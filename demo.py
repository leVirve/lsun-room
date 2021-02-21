import pathlib
import click
import cv2
import numpy as np
import torch
from loguru import logger

from datasets import sequence
from trainer import core

torch.backends.cudnn.benchmark = True


class Predictor:

    def __init__(self, weight_path):
        self.model = core.LayoutSeg.load_from_checkpoint(weight_path, backbone='resnet101')
        self.model.freeze()
        self.model.cuda()

    @torch.no_grad()
    def feed(self, image: torch.Tensor, alpha=.4) -> np.ndarray:
        _, outputs = self.model(image.unsqueeze(0).cuda())
        label = core.label_as_rgb_visual(outputs.cpu()).squeeze(0)
        blend_output = (image / 2 + .5) * (1 - alpha) + (label * alpha)
        return blend_output.permute(1, 2, 0).numpy()


@click.group()
def cli():
    pass


@cli.command()
@click.option('--path', type=click.Path(exists=True))
@click.option('--weight', type=click.Path(exists=True))
@click.option('--image_size', default=320, type=int)
def image(path, weight, image_size):
    logger.info('Press `q` to exit the sequence inference.')
    predictor = Predictor(weight_path=weight)
    images = sequence.ImageFolder(image_size, path)

    for image, shape, _ in images:
        output = cv2.resize(predictor.feed(image), shape)
        cv2.imshow('layout', output[..., ::-1])
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break


@cli.command()
@click.option('--path', type=click.Path(exists=True))
@click.option('--weight', type=click.Path(exists=True))
@click.option('--image_size', default=320, type=int)
@click.option('--cat_visual', is_flag=True)
@click.option('--output_folder', default='output/')
def save_result(path, weight, image_size, cat_visual, output_folder):
    output_folder = pathlib.Path(output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)

    predictor = Predictor(weight_path=weight)
    images = sequence.ImageFolder(image_size, path)

    for image, shape, path in images:
        label = cv2.resize(predictor.feed(image, alpha=1.), shape)
        image = cv2.resize((image / 2 + .5).permute(1, 2, 0).numpy(), shape)
        if cat_visual:
            output = np.concatenate([image, label], axis=1)
        else:
            output = label
        output_path = output_folder / path.name
        cv2.imwrite(str(output_path), (output[..., ::-1] * 255).astype(np.uint8))
        logger.info(f'Write to {output_path}')


@cli.command()
@click.option('--device', default=0)
@click.option('--path', type=click.Path(exists=True))
@click.option('--weight', type=click.Path(exists=True))
@click.option('--image_size', default=320, type=int)
def video(device, path, weight, image_size):
    logger.info('Press `q` to exit the sequence inference.')
    predictor = Predictor(weight_path=weight)
    stream = sequence.VideoStream(image_size, path, device)

    for image in stream:
        output = cv2.resize(predictor.feed(image), stream.origin_size)
        cv2.imshow('layout', output[:, :, ::-1])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    cli()
