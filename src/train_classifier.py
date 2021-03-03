import sys
import os
from typing import List
import logging
import librosa
from numpy.core.numeric import tensordot 
os.environ['HYDRA_FULL_ERROR'] = '1'
from argparse import Namespace
import hydra
from omegaconf import DictConfig
from sklearn.preprocessing import minmax_scale
import flash
from flash.vision import ImageClassificationData, ImageClassifier
from pytorch_lightning.loggers import WandbLogger as Logger
from pytorch_lightning.callbacks import ModelCheckpoint
from PIL import Image
try:
    import networks
    from dataloaders import SpectrogramsDataModule
    from dataloaders.audiodataset import AudioDataset
except ImportError:
    from src import networks
    from src.dataloaders.audiodataset import AudioDataset
    from src.dataloaders import SpectrogramsDataModule


def path_to_label(path:str, format:str = 'index', classes:List=None) -> int:
    """Extract the class name from file path and return the corresponding class_idx in classes.

    Args:
        path (str): [description]
        format(str): 'index' or 'name'
        classes (List): [description]

    Returns:
        [int]: Class index
    """
    class_name = path.split(os.sep)[-2]
    try:
        if format == 'index':
            return classes.index(class_name)
        else:
            return class_name
    except ValueError:
        return -1


def load_filepaths(infile:str) -> List[str]:
    with open(infile, mode='r') as reader:
        return reader.read().splitlines()

def load_sample(file_path, nb_second=4, sr=16384, resize=False):
    window_length = sr*nb_second
    audio, _sr = librosa.load(file_path, sr=sr)
    if _sr != sr:
        audio = librosa.resample(audio, _sr, sr)
    if len(audio) >= window_length:
        audio = audio[0:window_length]
    else:
        audio = librosa.util.fix_length(audio, window_length)
    features = AudioDataset.audio_to_melspectrogram(audio, resize=resize)
    # features = Image.fromarray(features)
    # features = torch.from_numpy(features)
    features = minmax_scale(features, (0, 255))
    features = Image.fromarray(features)
    return features



def get_data(cfg):
    train_filepaths = load_filepaths(cfg['dataset']['train_path'])
    val_filepaths   = load_filepaths(cfg['dataset']['val_path'])
    test_filepaths  = load_filepaths(cfg['dataset']['test_path'])
    train_filepaths = [os.path.join(cfg['dataset']['root_dir'], x) for x in train_filepaths]
    val_filepaths   = [os.path.join(cfg['dataset']['root_dir'], x) for x in val_filepaths]
    test_filepaths  = [os.path.join(cfg['dataset']['root_dir'], x) for x in test_filepaths]
    CLASSES_NAMES = cfg['dataset']['classes_name']
    datamodule = ImageClassificationData.from_filepaths(
        train_filepaths= train_filepaths,
        train_labels   = [path_to_label(x, format="index", classes=CLASSES_NAMES) for x in train_filepaths],
        val_filepaths  = val_filepaths,
        val_labels     = [path_to_label(x, format="index", classes=CLASSES_NAMES) for x in val_filepaths],
        test_filepaths = test_filepaths,
        test_labels    = [path_to_label(x, format="index", classes=CLASSES_NAMES) for x in test_filepaths],
        batch_size     = cfg['dataset']['batch_size'],
        num_workers    = cfg['dataset']['num_workers'],
        loader = load_sample,
    )
    return datamodule


@hydra.main(config_path="configs", config_name="train_classifier")
def main(cfg: DictConfig) -> None:
    if cfg.get('debug', False):
        logger = Logger(project=cfg['project_name'], name=cfg['run_name'], tags=cfg['tags']+["debug"])
    else:
        logger = Logger(project=cfg['project_name'], name=cfg['run_name'], tags=cfg['tags'])


    platform = sys.platform.lower() 
    if platform == "darwin":
        root_ = "/Users/test/Documents/Projects/Master/"
        cfg['dataset']['root_dir'] = os.path.join(root_, "udem-birds/classes")
        cfg['dataset']['train_path'] = os.path.join(root_, "udem-birds/samples/train_list.txt")
        cfg['dataset']['val_path'] = os.path.join(root_, "udem-birds/samples/val_list.txt")
        cfg['dataset']['test_path'] = os.path.join(root_, "udem-birds/samples/test_list.txt")

    # datamodule = get_data(cfg)
    datamodule = SpectrogramsDataModule(config=cfg['dataset'])

    # 3. Build the model
    FLASH_MODELS = [
        "resnet18", 
        "resnet34", 
        "resnet50",
        "resnet101",
        "resnet152",
        "resnext50_32x4d",
        "resnext101_32x8d",
        "mobilenet_v2",
        "vgg11",
        "vgg13",
        "vgg16",
        "vgg19",
        "densenet121",
        "densenet169",
        "densenet161",
        "swav-imagenet",
    ]

    num_classes = len(cfg['dataset']['classes_name'])
    if "efficientnet" in cfg['backbone_network']:
        model = networks.EfficientNet.from_pretrained(cfg['backbone_network'], num_classes=num_classes)
    elif cfg['backbone_network'] in FLASH_MODELS:
        model = ImageClassifier(num_classes=num_classes, backbone=cfg['backbone_network'])
    else:
        raise NotImplementedError("Network not implemented")


    # 4. Create the trainer. Run once on data
    checkpoint_callback = ModelCheckpoint('./models-classifier', monitor='val_accuracy', verbose=True)
    trainer = flash.Trainer(
        logger=logger,
        gpus=cfg.get('gpus', 0),
        max_epochs=cfg.get('nb_epochs', 3),
        checkpoint_callback=checkpoint_callback,
    )
    logger.log_hyperparams(cfg)
    # 5. Fit the model
    logging.info("Training...")
    trainer.fit(model, datamodule=datamodule)
    # trainer.finetune(model, datamodule=datamodule, strategy="freeze")
    
    logging.info("Testing...")
    test_results = trainer.test(datamodule=datamodule)
    # 6. Save it!
    trainer.save_checkpoint("last_model.pt")


if __name__ == "__main__":
    main()