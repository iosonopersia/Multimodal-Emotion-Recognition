<h1 align='center'> Multimodal Emotion Recognition </h1>

This repository contains a project realized as part of the _Natural Language Processing_ exam of the [Master's degree in Artificial Intelligence, University of Bologna](https://corsi.unibo.it/2cycle/artificial-intelligence).

## Prerequisites

This project was developed in `Python3` and `pytorch`. Run the following command to install the prerequisites:

```bash
# Linux
pip install --no-cache -r ./requirements_linux.txt

# Windows
pip install --no-cache -r ./requirements.txt
```

Otherwise, you can build a ready-to-go virtual environment by running the following scripts from the project's folder:

```bash
# Linux
> ./scripts/build-venv.sh

# Windows
> .\scripts\build-venv.bat
```

### Download FFMPEG

Download `ffmpeg` from [here](https://ffmpeg.org/download.html). Follow online tutorials to install it correctly based on your OS.

### Download dataset

Now, you need to download and prepare the dataset. Run the following commands from the project's folder:

```bash
# Linux
> ./scripts/MELD_download. # Download dataset
> ./scripts/video2wav.sh # Extract audio

# Windows
> .\scripts\MELD_download.bat # Download dataset
> .\scripts\video2wav.bat # Extract audio
```

## Group members

|  Reg No.  |  Name     |  Surname  |     Email                              |    Username      |
| :-------: | :-------: | :-------: | :------------------------------------: | :--------------: |
|  1005278  | Ludovico  | Granata   | `ludovico.granata@studio.unibo.it`     | [_LudovicoGranata_](https://github.com/LudovicoGranata) |
|  973719  | Parsa     | Dahesh    | `parsa.dahesh@studio.unibo.it`         | [_ParsaD23_](https://github.com/ParsaD23) |
|  984854  | Simone    | Persiani  | `simone.persiani2@studio.unibo.it`     | [_iosonopersia_](https://github.com/iosonopersia) |

