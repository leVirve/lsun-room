import click
import cv2
import onegan
import torch
import torchvision.transforms as T
from PIL import Image

from trainer.賣扣老師 import build_resnet101_FCN

torch.backends.cudnn.benchmark = True


class Predictor:

    def __init__(self, input_size, weight=None):
        self.model = self.build_model(weight)
        self.colorizer = onegan.extension.Colorizer(
            colors=[
                [249, 69, 93], [255, 229, 170], [144, 206, 181],
                [81, 81, 119], [241, 247, 210]])
        self.transform = T.Compose([
            T.Resize(input_size),
            T.ToTensor(),
            T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
        ])

    def build_model(self, weight_path, joint_class=False):
        model = build_resnet101_FCN(pretrained=False, nb_classes=37, stage_2=True, joint_class=joint_class)
        weight = onegan.utils.export_checkpoint_weight(weight_path)
        model.load_state_dict(weight)
        model.eval()
        return model.cuda()

    @onegan.utils.timeit
    def process(self, raw):

        def _batched_process(batched_img):
            score, _ = self.model(onegan.utils.to_var(batched_img))
            _, output = torch.max(score, 1)

            image = (batched_img / 2 + .5)
            layout = self.colorizer.apply(output.data.cpu())
            return image * .6 + layout * .4

        img = Image.fromarray(raw)
        batched_img = self.transform(img).unsqueeze(0)
        canvas = _batched_process(batched_img)
        result = canvas.squeeze().permute(1, 2, 0).numpy()
        return cv2.resize(result, (raw.shape[1], raw.shape[0]))


@click.command()
@click.option('--device', default=0)
@click.option('--video', type=click.Path(exists=True))
@click.option('--weight', type=click.Path(exists=True))
@click.option('--input_size', default=(320, 320), type=(int, int))
def main(device, video, weight, input_size):

    demo = Predictor(input_size, weight=weight)

    reader = video if video else device
    cap = cv2.VideoCapture(reader)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        output = demo.process(frame[:, :, ::-1])

        cv2.imshow('layout', output[:, :, ::-1])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
