import SimpleITK
import numpy as np
import cv2
import argparse


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--file", type=str, required=True)
    arg_parser.add_argument("--interval", type=int, default=100)
    arg_parser.add_argument("--size", nargs="+", default=[512, 512])
    arg_parser.add_argument("--abs", type=bool, default=False)

    args = arg_parser.parse_args()
    filepath: str = args.file
    interval: int = int(args.interval)
    size: tuple[int, ...] = tuple([int(x) for x in args.size])
    use_abs: bool = args.abs in [True, "True", "Y", "Yes", "y", "yes"]

    scan = SimpleITK.ReadImage(filepath)
    scan = SimpleITK.GetArrayFromImage(scan)
    if use_abs:
        scan = np.abs(scan)
    scan = (scan - np.min(scan)) / (np.max(scan) - np.min(scan))

    slice_count = scan.shape[0]
    stop = False
    i = (np.argmax(np.max(scan, axis=(1, 2)), axis=0) - 8) % slice_count
    while not stop:
        current_slice = scan[i]
        current_slice = cv2.resize(current_slice, size, interpolation=cv2.INTER_NEAREST)

        cv2.imshow("Current Slice", current_slice)
        user_input = cv2.waitKey(interval)

        if user_input in [13, 27]:
            stop = True
        i = (i + 1) % slice_count


if __name__ == "__main__":
    main()
