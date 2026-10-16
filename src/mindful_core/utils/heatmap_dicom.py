import numpy as np
import pydicom
import pydicom.uid
# noinspection PyProtectedMember
from pydicom._storage_sopclass_uids import SegmentationStorage
from pydicom_seg import writer_utils
from pydicom_seg.template import from_dcmqi_metainfo
from pydicom_seg.dicom_utils import CodeSequence, DimensionOrganizationSequence
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

from mindful_core.utils.misc import try_load_json


class HeatmapDataset(pydicom.Dataset):
    """
        Based on SegmentationDataset from pydicom-seg
        """

    def __init__(self,
                 rows: int,
                 columns: int,
                 reference_dicom: pydicom.Dataset | None,
                 max_fractional_value: int = 255,
                 manufacturer_model_name: str = "Mindful"
                 ):
        super().__init__()

        self._reference_dicom = reference_dicom

        writer_utils.import_hierarchy(
            target=self,
            reference=reference_dicom,
            import_patient=True,
            import_study=True,
            import_frame_of_reference=True,
            import_series=False,
            # import_series=True,
            import_charset=True
        )

        self.preamble = b"\0" * 128
        self.SpecificCharacterSet = "ISO_IR 100"
        self.SOPClassUID = SegmentationStorage
        self.SOPInstanceUID = pydicom.uid.generate_uid()
        init_file_meta(self)

        # region Generate series data
        # General Series module
        self.Modality = "SEG"
        self.SeriesInstanceUID = pydicom.uid.generate_uid()
        # self.SeriesInstanceUID = reference_dicom.SeriesInstanceUID
        # self.SeriesNumber = 1

        # Generate SOP and Series and General Image timestamps
        timestamp = datetime.now()
        self.InstanceCreationDate = timestamp.strftime("%Y%m%d")
        self.InstanceCreationTime = timestamp.strftime("%H%M%S.%f")
        self.SeriesDate = self.InstanceCreationDate
        self.SeriesTime = self.InstanceCreationTime
        self.ContentDate = self.InstanceCreationDate
        self.ContentTime = self.InstanceCreationTime
        # endregion

        # Enhanced General Equipment module
        # http://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.5.2.html#table_C.7-8b
        self.Manufacturer = "heds"
        self.ManufacturerModelName = manufacturer_model_name
        self.DeviceSerialNumber = "0"
        self.SoftwareVersions = "1.5"

        # Image Pixel module
        self.NumberOfFrames, self.Rows, self.Columns = (0, rows, columns)

        # Segmentation Image module
        # http://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.20.2.html#table_C.8.20-2
        self.ImageType = ["DERIVED", "PRIMARY"]
        self.InstanceNumber = "1"
        self.ContentLabel = "SEGMENTATION"
        self.ContentDescription = ""
        self.ContentCreatorName = ""
        self.SamplesPerPixel = 1
        self.PhotometricInterpretation = "MONOCHROME2"
        self.PixelRepresentation = 0
        self.LossyImageCompression = "00"
        self.SegmentSequence = pydicom.Sequence()

        self.SegmentationType = "FRACTIONAL"
        # Fractional segmentations are always 8-bit unsigned
        self.SegmentationFractionalType = "PROBABILITY"
        self._validate_max_fractional_value(max_fractional_value)
        self.MaximumFractionalValue = max_fractional_value
        self.BitsAllocated = 8
        self.BitsStored = 8
        self.HighBit = 7

        self.SharedFunctionalGroupsSequence = pydicom.Sequence([pydicom.Dataset()])
        self.PerFrameFunctionalGroupsSequence = pydicom.Sequence()

        # Un-initialized attributes
        self.PixelData: bytes | None = None
        self.DimensionOrganizationSequence: pydicom.Sequence | None = None
        self.DimensionIndexSequence: pydicom.Sequence | None = None
        self.ReferencedSeriesSequence: pydicom.Sequence | None = None

        self._heatmaps: list[np.ndarray] = []

    def add_heatmap(self,
                    image: np.ndarray,
                    referenced_segment: int,
                    referenced_frames: list[pydicom.Dataset],
                    update_pixel_data: bool = True,
                    ) -> None:
        self._validate_segment_number(referenced_segment)
        if len(image) != len(referenced_frames):
            raise ValueError("Number of frames did not match, got {} and {}".format(len(image), len(referenced_frames)))

        self._append_pixel_data(image, update_pixel_data)

        segment_identification = pydicom.Dataset()
        segment_identification.ReferencedSegmentNumber = referenced_segment
        for frame_idx in range(len(image)):
            self.add_frame(frame_idx,
                           referenced_segment,
                           referenced_frames[frame_idx],
                           segment_identification)

    def add_frame(self,
                  frame_idx: int,
                  referenced_segment: int,
                  referenced_image: pydicom.Dataset,
                  segment_identification: pydicom.Dataset):
        self.add_instance_reference(referenced_image)

        # region DerivationCodeSequence
        ref = pydicom.Dataset()
        ref.ReferencedSOPClassUID = referenced_image.SOPClassUID
        ref.ReferencedSOPInstanceUID = referenced_image.SOPInstanceUID
        ref.PurposeOfReferenceCodeSequence = CodeSequence(value="121322",
                                                          scheme_designator="DCM",
                                                          meaning="Source image for image processing operation")

        derivation_image = pydicom.Dataset()
        derivation_image.SourceImageSequence = pydicom.Sequence()
        derivation_image.SourceImageSequence.append(ref)
        derivation_image.DerivationCodeSequence = CodeSequence(value="113076",
                                                               scheme_designator="DCM",
                                                               meaning="Segmentation")
        derivation_image_sequence = pydicom.Sequence([derivation_image])
        # endregion

        # region FrameContentSequence
        frame_content_sequence = pydicom.Dataset()
        # idx + 1 because "index values are defined to start from 1 and monotonically increase by 1"
        frame_content_sequence.DimensionIndexValues = [referenced_segment, frame_idx + 1]
        # endregion

        # region PlanePositionSequence
        plane_position_sequence = pydicom.Dataset()
        origin = get_image_position_patient(referenced_image)
        frame_position = frame_index_to_position(origin, self.slice_thickness, frame_idx, format_result=True)
        plane_position_sequence.ImagePositionPatient = frame_position
        # endregion

        heatmap_frame_item = pydicom.Dataset()
        heatmap_frame_item.SegmentIdentificationSequence = [segment_identification]
        heatmap_frame_item.DerivationImageSequence = derivation_image_sequence
        heatmap_frame_item.FrameContentSequence = [frame_content_sequence]
        heatmap_frame_item.PlanePositionSequence = [plane_position_sequence]
        self.add_heatmap_frame_item(heatmap_frame_item)

    def add_instance_reference(self, dataset: pydicom.Dataset) -> bool:
        if "ReferencedSeriesSequence" not in self:
            self.ReferencedSeriesSequence = pydicom.Sequence()

        for series_item in self.ReferencedSeriesSequence:
            if series_item.SeriesInstanceUID != dataset.SeriesInstanceUID:
                continue

            for instance_item in series_item.ReferencedInstanceSequence:
                if instance_item.ReferencedSOPInstanceUID == dataset.SOPInstanceUID:
                    return False

            # Series found, but instance is missing
            break
        else:
            # Series not yet referenced, create a new series item
            series_item = pydicom.Dataset()
            series_item.SeriesInstanceUID = dataset.SeriesInstanceUID
            series_item.ReferencedInstanceSequence = pydicom.Sequence([])
            self.ReferencedSeriesSequence.append(series_item)

        # Instance not yet referenced, create a new instance item
        instance_item = pydicom.Dataset()
        instance_item.ReferencedSOPClassUID = dataset.SOPClassUID
        instance_item.ReferencedSOPInstanceUID = dataset.SOPInstanceUID
        series_item.ReferencedInstanceSequence.append(instance_item)

        return True

    def add_heatmap_frame_item(self, heatmap_frame_item: pydicom.Dataset):
        expected_fields = ["SegmentIdentificationSequence",
                           "DerivationImageSequence",
                           "FrameContentSequence",
                           "PlanePositionSequence"]
        missing_fields = [field for field in expected_fields if field not in heatmap_frame_item]
        if len(missing_fields) > 0:
            raise RuntimeError("Heatmap frame item is missing the following attributes: {}".format(missing_fields))
        self.PerFrameFunctionalGroupsSequence.append(heatmap_frame_item)

    # region Pixel data
    def _append_pixel_data(self, data: np.ndarray, update_pixel_data: bool) -> None:
        """

        :param data: A 3D array with shape [depth, height, width]
        :param update_pixel_data: If True, updates the pixel data of the dataset
        """
        if len(data.shape) != 3:
            raise ValueError("Invalid frame data shape ({}), expecting 3D images".format(data.shape))

        if (data.shape[1] != self.Rows) or (data.shape[2] != self.Columns):
            raise ValueError("Invalid frame data shape, expecting {}x{} images".format(self.Rows, self.Columns))

        self.NumberOfFrames += data.shape[0]

        data = data.astype(np.float32) if not np.issubdtype(data.dtype, np.floating) else data
        data = (data - data.min()) / (data.max() - data.min())
        data *= self.MaximumFractionalValue
        data = data.astype(np.uint8)
        data = data.ravel()
        self._heatmaps.append(data)

        if update_pixel_data:
            self.update_pixel_data()

    def update_pixel_data(self):
        raw_pixel_data = np.concatenate(self._heatmaps)
        self.PixelData = raw_pixel_data.tobytes()

    # endregion

    def add_dimension_organization(self, dim_organization: DimensionOrganizationSequence = None):
        if "DimensionOrganizationSequence" not in self:
            self.DimensionOrganizationSequence = pydicom.Sequence()
            self.DimensionIndexSequence = pydicom.Sequence()

        if dim_organization is None:
            dim_organization = DimensionOrganizationSequence()
            dim_organization.add_dimension("ReferencedSegmentNumber", "SegmentIdentificationSequence")
            dim_organization.add_dimension("ImagePositionPatient", "PlanePositionSequence")

        for item in self.DimensionOrganizationSequence:
            if item.DimensionOrganizationUID == dim_organization[0].DimensionOrganizationUID:
                raise ValueError("Dimension organization with UID {} already exists".
                                 format(item.DimensionOrganizationUID))

        item = pydicom.Dataset()
        item.DimensionOrganizationUID = dim_organization[0].DimensionOrganizationUID
        self.DimensionOrganizationSequence.append(item)
        self.DimensionIndexSequence.extend(dim_organization)

    # region Initialization / Validation
    @staticmethod
    def _validate_max_fractional_value(max_fractional_value: int):
        if max_fractional_value < 1 or max_fractional_value > 255:
            raise ValueError("Invalid maximum fractional value for 8-bit unsigned int data")

    def _validate_segment_number(self, segment_number: int):
        if segment_number not in self.segment_numbers:
            raise ValueError("Invalid segment number ({}). Expected one of: {}".
                             format(segment_number, self.segment_numbers))

    def validate_dataset(self, template_path: str):
        template: dict = try_load_json(template_path, "Heatmap DICOM template")

        expected_attributes = []
        for group_name, group_attributes in template.items():
            for attribute in group_attributes:
                if isinstance(attribute, str):
                    expected_attributes.append(attribute)
                elif isinstance(attribute, dict):
                    expected_attributes += list(attribute.keys())

        expected_attributes = list(set(expected_attributes))
        missing_attributes = [attribute for attribute in expected_attributes
                              if (attribute not in self) or (getattr(self, attribute) is None)]
        if len(missing_attributes) > 0:
            raise RuntimeError("Dataset is missing the following attributes: {}".format(missing_attributes))

    # endregion

    @property
    def segment_numbers(self) -> list[int]:
        return [segment.SegmentNumber for segment in self.SegmentSequence]

    @property
    def slice_thickness(self) -> float:
        return get_slice_thickness(self._reference_dicom)


class HeatmapWriter(object):
    def __init__(self, template: str | Path | dict | pydicom.Dataset):
        if isinstance(template, (str, Path)):
            template = try_load_json(template, "Heatmap DICOM template")

        if isinstance(template, dict):
            template = from_dcmqi_metainfo(template)
        self.template = template

    def run(self,
            heatmaps: list[np.ndarray] | np.ndarray,
            reference_dicom_path: str | Path
            ) -> HeatmapDataset:
        # region Prepare/validate data
        if isinstance(heatmaps, np.ndarray):
            heatmaps = [heatmaps]
        self._validate_heatmaps(heatmaps)
        ref_heatmap = heatmaps[0]

        reference_dicom_path = Path(reference_dicom_path)
        reference_dicom = pydicom.dcmread(reference_dicom_path)

        referenced_frames_path = reference_dicom_path.parent / reference_dicom_path.stem
        if referenced_frames_path.exists():
            referenced_frames = [pydicom.dcmread(filepath.as_posix())
                                 for filepath in referenced_frames_path.glob("*.dcm")]
        else:
            referenced_frames = unstack_dicom_frames(reference_dicom)
            referenced_frames_path.mkdir()

        for i, referenced_frame in enumerate(referenced_frames):
            referenced_frame.save_as(referenced_frames_path / "{:03d}.dcm".format(i))
        # endregion

        rows, columns = ref_heatmap.shape[1:]
        result = HeatmapDataset(rows=rows,
                                columns=columns,
                                reference_dicom=reference_dicom,
                                max_fractional_value=255)
        result.add_dimension_organization()
        writer_utils.copy_segmentation_template(
            target=result,
            template=self.template,
            segments=self.declared_segments,
            skip_missing_segment=False,
        )
        result.InstanceNumber = self.template.InstanceNumber
        spacing = get_image_spacing(reference_dicom)
        raw_image_orientation_patient = get_image_orientation_patient(reference_dicom) or (0.0, 0.0, 0.0)
        image_orientation_patient = format_dcm_float(raw_image_orientation_patient)
        set_shared_functional_groups_sequence(target=result,
                                              spacing=spacing,
                                              image_orientation_patient=image_orientation_patient)

        for segment_number, heatmap in zip(self.declared_segments, heatmaps):
            heatmap: np.ndarray
            result.add_heatmap(heatmap,
                               referenced_segment=segment_number,
                               referenced_frames=referenced_frames,
                               update_pixel_data=False)
        result.update_pixel_data()

        result.SegmentsOverlap = "NO"

        return result

    # region Validate data
    @staticmethod
    def _validate_heatmaps(heatmaps: list[np.ndarray]):
        if not isinstance(heatmaps, (list, tuple)):
            raise ValueError("You must provide either a list or a tuple of images, got {}.".format(type(heatmaps)))

        if len(heatmaps) == 0:
            raise ValueError("You must provide at least one image.")

        if not all([isinstance(x, np.ndarray) for x in heatmaps]):
            raise ValueError("At least one element of your list/tuple was not a np.ndarray")

        ref_shape = heatmaps[0].shape
        if not all([ref_shape == heatmap.shape for heatmap in heatmaps]):
            raise ValueError("Currently only supporting when all images have the same size")

    # endregion

    @property
    def declared_segments(self) -> set[int]:
        return set([segment.SegmentNumber for segment in self.template.SegmentSequence])


def init_file_meta(dataset: pydicom.Dataset) -> None:
    if pydicom.__version_info__[0] == "1":
        dataset.file_meta = pydicom.Dataset()
    else:
        dataset.file_meta = pydicom.dataset.FileMetaDataset()
    dataset.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    dataset.file_meta.MediaStorageSOPInstanceUID = dataset.SOPInstanceUID
    dataset.file_meta.MediaStorageSOPClassUID = dataset.SOPClassUID
    pydicom.dataset.validate_file_meta(dataset.file_meta)

    # Fix missing FileMetaInformationGroupLength. It is added by `pydicom` when saving with
    # `write_as_original=False`, but this can be a dangerous pitfall if not done correctly
    if "FileMetaInformationGroupLength" not in dataset.file_meta:
        # See: https://github.com/pydicom/pydicom/blob/e8de9d31fc97e1162441adf4bd2742b82149ce18/pydicom
        # /filewriter.py#L645-L736
        buffer = pydicom.filewriter.DicomBytesIO()
        buffer.is_little_endian = True
        buffer.is_implicit_VR = False
        pydicom.filewriter.write_dataset(buffer, dataset.file_meta)
        dataset.file_meta.FileMetaInformationGroupLength = buffer.tell()


def unstack_dicom_frames(dataset: pydicom.Dataset, pixel_type=np.uint16) -> list[pydicom.Dataset]:
    single_frame_datasets = []
    pixel_data = dataset.pixel_array.astype(pixel_type)

    origin = get_image_position_patient(dataset)
    if origin is None:
        origin = (0, 0, 0)

    additional_tags_to_copy = ["SOPClassUID",
                               "Rows", "Columns",
                               "ImageType", "BitsAllocated", "BitsStored", "HighBit",
                               "ContentLabel", "ContentDescription", "ContentCreatorName",
                               "SamplesPerPixel", "PhotometricInterpretation", "PixelRepresentation",
                               "LossyImageCompression"]

    for i in range(dataset.NumberOfFrames):
        single_frame_dataset = pydicom.Dataset()

        writer_utils.import_hierarchy(
            target=single_frame_dataset,
            reference=dataset,
            import_patient=True,
            import_study=True,
            import_frame_of_reference=True,
            import_series=True,
            import_charset=True
        )

        single_frame_dataset.preamble = b"\0" * 128
        for tag in additional_tags_to_copy:
            if tag in dataset:
                setattr(single_frame_dataset, tag, getattr(dataset, tag))

        sop_instance_uid = pydicom.uid.generate_uid(prefix=None)
        single_frame_dataset.SOPInstanceUID = sop_instance_uid
        init_file_meta(single_frame_dataset)

        frame = pixel_data[i]
        single_frame_dataset.PixelData = frame.tobytes()

        slice_thickness = get_slice_thickness(dataset)
        image_position_patient = frame_index_to_position(origin, slice_thickness, i, format_result=True)
        detector_information_sequence = pydicom.Dataset()
        detector_information_sequence.ImagePositionPatient = image_position_patient
        single_frame_dataset.DetectorInformationSequence = [detector_information_sequence]
        single_frame_dataset.ImagePositionPatient = image_position_patient
        single_frame_dataset.SliceLocation = image_position_patient[0]
        single_frame_dataset.InstanceNumber = i

        single_frame_datasets.append(single_frame_dataset)

    return single_frame_datasets


def get_attribute_recursively(dataset: Iterable, attribute_name: str):
    if hasattr(dataset, attribute_name):
        return getattr(dataset, attribute_name)

    for element in dataset:
        if element.VR == "SQ":
            for item in element:
                result = get_attribute_recursively(item, attribute_name)
                if result is not None:
                    return result

    return None


def get_image_position_patient(dataset: Iterable) -> Sequence[float] | tuple[float, float, float] | None:
    return get_attribute_recursively(dataset, "ImagePositionPatient")


def get_image_orientation_patient(dataset: Iterable) -> tuple[float, float, float, float, float, float]:
    return get_attribute_recursively(dataset, "ImageOrientationPatient")


def get_image_spacing(dataset: Iterable) -> tuple[float, float, float]:
    z_spacing: float = get_attribute_recursively(dataset, "SpacingBetweenSlices")
    y_spacing, x_spacing = get_attribute_recursively(dataset, "PixelSpacing")
    return z_spacing, y_spacing, x_spacing


def get_slice_thickness(dataset: pydicom.Dataset) -> float:
    result = get_attribute_recursively(dataset, "SliceThickness")
    if result is not None:
        return result

    result = get_attribute_recursively(dataset, "SpacingBetweenSlices")
    if result is not None:
        return result

    raise ValueError


def frame_index_to_position(origin: tuple[float, float, float],
                            slice_thickness: float,
                            frame_index: int,
                            format_result: bool) -> tuple[float, float, float] | tuple[str, str, str]:
    pos_z, pos_x, pos_y = origin
    pos_z += slice_thickness * frame_index
    result = (pos_z, pos_x, pos_y)
    if format_result:
        result = format_dcm_float(result)
    return result


# noinspection PyPep8Naming
def set_shared_functional_groups_sequence(target: pydicom.Dataset,
                                          spacing: tuple[float, float, float],
                                          image_orientation_patient: list[str]
                                          ) -> None:
    sx, sy, sz = spacing

    dataset = pydicom.Dataset()
    dataset.PixelMeasuresSequence = [pydicom.Dataset()]
    dataset.PixelMeasuresSequence[0].PixelSpacing = [f"{sy:e}", f"{sx:e}"]
    dataset.PixelMeasuresSequence[0].SliceThickness = f"{sz:e}"
    dataset.PixelMeasuresSequence[0].SpacingBetweenSlices = f"{sz:e}"
    dataset.PlaneOrientationSequence = [pydicom.Dataset()]
    dataset.PlaneOrientationSequence[0].ImageOrientationPatient = image_orientation_patient

    target.SharedFunctionalGroupsSequence = pydicom.Sequence([dataset])


def format_dcm_float(values: Sequence[float]) -> list[str]:
    return ["{:e}".format(x) for x in values]
