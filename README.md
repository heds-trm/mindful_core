# Mindful-Core
Medical ImagiNg Data FUsion Lab

## Project Structure
Below you will find an overview of the structure of the project.

### main.py
This is the main entry point of the project and will start a series of experiments based on the configuration file. (see [Experiments](#experiments))

### Experiments
To automate experiments and fold-by-fold experimentation, experiments use the following hierachy:
`ExperimentSeries > Experiment > ExperimentRound > ExperimentStage`

#### 1. ExperimentSeries
*ExperimentSeries is a set of Experiment (N=number of experiments described in the configuration).*

[ExperimentSeries](experiments/experiment_series.py) uses the main configuration to automatically chain experiments described in the configuration. Its role is both to iterate through the experiments and dynamically update the configs of these experiments. For example, ExperimentSeries will share the same root folder for all its experiments which will be logged within their own separate subfolders in this root folder. ExperimentSeries will make sure seeds used by Experiments are the same.

#### 2. Experiment
*Experiment is a set of ExperimentRound (N=number of folds).*

[Experiment](experiments/experiment.py) will start experiment rounds which can train, validate, test, ... individual models. The key difference between Experiment and ExperimentRound is that Experiment will run as many ExperimentRound as needed based on the number of folds in the experiment. Experiment will make sure the seed used by each ExperimentRound (fold) is different.

#### 3. ExperimentRound
*ExperimentRound is a set of ExperimentStage (N=number of stages in config).*

__[ExperimentRound](experiments/experiment_round.py) is the heart of this project.__ It will create models, data pipelines, load data folds, ... based on the configuration it receives. Most commonly, ExperimentRound will run the training and test stages, as well as run any interpretability or additional stages. 

#### 4. ExperimentStage
[ExperimentStage](experiments/experiment_stage.py) is only a registry of stages the ExperimentRound must perform and contain very little data by themselves. This class mainly exists for allowing future versions to add data to configure stages.

### Models

#### 1. MindfulModule
[MindfulModule](models/module.py) is an abstract class and the root of any model class supported by this project. It will mainly configure handles for the loss function and optimizers.
By themselves, MindfulModule do not define any architecture, task or expected inputs/outputs and can therefore be derived for any use.

Classes that inherit from MindfulModule also benefit from being added to a registry, which allows the project to automatically recognize any class/model imported in the project as a valid model to create or load.

#### 2. ModelOutput
[ModelOutput](models/model_output.py) defines all the supported model outputs (classification, segmentation, bounding boxes, ...) and gives an interface to help manipulate these outputs.

#### 3. AbstractClassifier
[AbstractClassifier](models/classification/abstract_classifier.py) is an abstract class and the root of any classification model. 
AbstractClassifier defines the base classification by itself, depending on the arguments used to build the instance.
The architecture and the forward function are left to subclasses. The forward function of subclasses must return a valid ClassifierOutput instance (see [ModelOutput](#modeloutput)).

The classification models available in this project are:
- [MonaiClassifier](models/classification/monai_classifier.py): a basic bridge between the Mindful project and MONAI. See the `make_monai_classifier` method for supported architectures.
- [TargetModel](models/classification/target_model.py): a classification model in two parts. The first part is any encoder that output a 1D representation and the second an MLP projecting the representation to class logits.
- [EnsembleClassifier](models/classification/ensemble/ensemble_classifier.py): an ensemble model aggregating the logits of multiple models.

### Data
### Analysis
### Scripts
### Utils