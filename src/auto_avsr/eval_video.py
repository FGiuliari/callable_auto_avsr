import os
import torch
import torchaudio
import torchvision
import os
from .lightning import ModelModule
from .datamodule.transforms import AudioTransform, VideoTransform
from pathlib import Path

# %%

# import logging
#
# L = logging.getLogger()
# L.setLevel(logging.DEBUG)


class InferencePipeline(torch.nn.Module):
    def __init__(self, args, ckpt_path, detector="retinaface"):
        super(InferencePipeline, self).__init__()
        self.modality = args.modality
        print(self.modality)
        if self.modality == "audio":
            self.audio_transform = AudioTransform(subset="test")
        elif self.modality == "video":
            if detector == "mediapipe":
                from .preparation.detectors.mediapipe.detector import LandmarksDetector
                from .preparation.detectors.mediapipe.video_process import VideoProcess

                self.landmarks_detector = LandmarksDetector()
                self.video_process = VideoProcess(convert_gray=False)
            elif detector == "retinaface":
                from .preparation.detectors.retinaface.detector import LandmarksDetector
                from .preparation.detectors.retinaface.video_process import VideoProcess

                self.landmarks_detector = LandmarksDetector(device="cuda:0")
                self.video_process = VideoProcess(convert_gray=False)
            self.video_transform = VideoTransform(subset="test")

        ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        self.modelmodule = ModelModule(args)
        self.modelmodule.model.load_state_dict(ckpt, strict=False)
        self.modelmodule.eval()
        self.modelmodule.cuda()

    def load_video(self, data_filename):
        return torchvision.io.read_video(data_filename, pts_unit="sec")[0].numpy()

    def forward(self, data_filename):
        data_filename = os.path.abspath(data_filename)
        assert os.path.isfile(
            data_filename
        ), f"data_filename: {data_filename} does not exist."

        if self.modality == "audio":
            audio, sample_rate = self.load_audio(data_filename)
            audio = self.audio_process(audio, sample_rate)
            audio = audio.transpose(1, 0)
            audio = self.audio_transform(audio)
            with torch.no_grad():
                transcript = self.modelmodule(audio)

        if self.modality == "video":
            video = self.load_video(data_filename)
            print("calculating landmarks")
            landmarks = self.landmarks_detector(video)
            print("processing video")
            video = self.video_process(video, landmarks)
            video = torch.tensor(video)
            video = video.permute((0, 3, 1, 2))
            print("doing video transform")
            video = self.video_transform(video)
            with torch.no_grad():
                print("doing inference")
                transcript = self.modelmodule(video.cuda())

        return transcript

    def load_audio(self, data_filename):
        waveform, sample_rate = torchaudio.load(data_filename, normalize=True)
        return waveform, sample_rate

    def load_video(self, data_filename):
        return torchvision.io.read_video(data_filename, pts_unit="sec")[0].numpy()

    def audio_process(self, waveform, sample_rate, target_sample_rate=16000):
        if sample_rate != target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, target_sample_rate
            )
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform


# %% [markdown]
# ## 2. Download a video from web

# %%

# %% [markdown]
# ## 3. VSR inference

# %% [markdown]
# ### 3.1 Download a pre-trained model


def main():
    import argparse

    parser = argparse.ArgumentParser(
        prog="vsr_baseline_autoAVSR",
        description="What the program does",
    )
    parser.add_argument("filename", type=Path)  # positional argument
    parser.add_argument("output_folder", type=Path, default=".")
    args = parser.parse_args()
    model_path = "/home/fgiuliari/Documents/Projects/Industrial-Projects/Meetween/2026_ECCV_VSR_BENCH/baselines/auto_avsr/ckpts/vsr_trlrs2lrs3vox2avsp_base.pth"

    # %% [markdown]
    # ### 3.2 Initialize VSR pipeline

    #
    # %%
    setattr(args, "modality", "video")
    pipeline = InferencePipeline(args, model_path, detector="mediapipe")

    # %% [markdown]
    # ### 3.3 Run inference

    # %%
    transcript = pipeline(args.filename)
    print(transcript)

    output_folder: Path = args.output_folder
    output_folder.mkdir(exist_ok=True, parents=True)

    file_name = args.filename.stem

    with open(output_folder / f"{file_name}.txt", "w") as f:
        print(transcript, file=f)


if __name__ == "__main__":
    main()
