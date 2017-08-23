import click
import torch

from dataset import ImageFolderDataset
from net import *

torch.backends.cudnn.benchmark = True

@click.command()
@click.option('--name', type=str)
@click.option('--dataset_root', default='../SemanticTransfer')
@click.option('--image_size', default=(404, 404), type=(int, int))
@click.option('--epochs', default=20, type=int)
@click.option('--batch_size', default=1, type=int)
@click.option('--workers', default=6, type=int)
@click.option('--resume', type=click.Path(exists=True))
def main(name, dataset_root, image_size, epochs, batch_size, workers, resume):

	print('===> Prepare data loader')
	dataset_args = {'root':dataset_root, 'target_size': image_size}
	loader_args = {'num_workers': workers, 'pin_memory': True}

	train_loader = torch.utils.data.DataLoader(
		dataset=ImageFolderDataset(phase='train', **dataset_args),
		batch_size=batch_size, shuffle=True, **loader_args
		)
	validate_loader = torch.utils.data.DataLoader(
		dataset=ImageFolderDataset(phase='validate', **dataset_args),
		batch_size=batch_size, **loader_args
		)
	test_loader = torch.utils.data.DataLoader(
		dataset=ImageFolderDataset(phase='test', **dataset_args),
		batch_size=batch_size, **loader_args
		)

	print('===> Prepare model')

	net = StageNet(name='FCN32s-1', pretrained=True)
	print('===> Start training')
	net.train(train_loader=train_loader,
		validate_loader=validate_loader,
		epochs=epochs)
	net.evaluate(data_loader=validate_loader, prefix='')
	net.predict(data_loader=test_loader, name='FCN32s_seg_layout')

def stage_two(name, dataset_root, image_size, epochs, batch_size, workers, resume):

	print('===> Prepare data loader')
	dataset_args = {'root':dataset_root, 'target_size': image_size}
	loader_args = {'num_workers': workers, 'pin_memory': True}

	train_loader = torch.utils.data.DataLoader(
		dataset=ImageFolderDataset(phase='train', **dataset_args),
		batch_size=batch_size, shuffle=True, **loader_args
		)
	validate_loader = torch.utils.data.DataLoader(
		dataset=ImageFolderDataset(phase='validate', **dataset_args),
		batch_size=batch_size, **loader_args
		)
	test_loader = torch.utils.data.DataLoader(
		dataset=ImageFolderDataset(phase='test', **dataset_args),
		batch_size=batch_size, **loader_args
		)

	print('===> Prepare model')

	net = StageNet(name='FCN32s-1', pretrained=True)
	print('===> Start training')
	net.train(train_loader=train_loader,
		validate_loader=validate_loader,
		epochs=epochs)
	net.evaluate(data_loader=validate_loader, prefix='')
#	net.predict(data_loader=test_loader, name='FCN32s_seg_layout')
if __name__ == '__main__':
	main()
