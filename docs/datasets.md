# Datasets

## NZZ black-letter ground truth (Zenodo)

| | |
|---|---|
| Record | https://doi.org/10.5281/zenodo.3333627 (published 2019-07-12) |
| Content | 167 NZZ front pages, one per year, 1780–1947 (Fraktur period) |
| Annotation | PAGE XML, manually corrected in Transkribus. ~100 pages corrected on word and line level, the rest on line level only. Read text **only from `TextLine/TextEquiv`**. |
| Archive | `NZZ-black-letter-ground-truth-master.zip`, 472 003 256 bytes |
| md5 | `7b9f0b42a65ff363755419496496638b` |

### Download

```
python scripts/download_zenodo.py            # default: data/zenodo/
```

The script reads file list and checksum from the Zenodo API, downloads, verifies the md5 (aborts on mismatch) and extracts to `data/zenodo/extracted/`. The data directory is git-ignored; only `data/zenodo/README.md` is tracked.

### Folder layout (after extraction)

```
data/zenodo/extracted/NZZ-black-letter-ground-truth-master/
├── img/                                  # 167 page scans (*.tif, ~423 MB)
├── xml/
│   ├── NZZ_groundtruth/                  # corrected ground truth (167 PAGE XML)
│   ├── ABBYY_FineReader_Server11/        # raw ABBYY output (167)
│   └── ABBYY_FineReader_XIX/             # raw ABBYY output, XIX model (167)
├── test-set-filenames.txt                # 17 pages (original authors' test set)
├── word-based-transcriptions-filenames.txt  # 135 pages with word-level correction
├── README.md
└── LICENSE.txt
```

File names follow `nzz_<YYYYMMDD>_0_0_a1_p1_1` (images prefixed with a running number).

### License

The two sources disagree, so plan for the stricter one:

- Zenodo record metadata: CC BY 4.0
- `LICENSE.txt` / README in the archive: CC BY-NC 4.0 ("can be used for academic purposes")

Use is academic (HSLU DSPRO2), non-commercial. Do not redistribute the data in this repo.

### Citation

Ströbel, P. & Clematide, S. (2019). *Improving OCR of Black Letter in Historical Newspapers: The Unreasonable Effectiveness of HTR Models on Low-Resolution Images.* Proceedings of the Digital Humanities 2019 (DH2019).

```bibtex
@inproceedings{clematide-stroebel-2019,
  author = "Ströbel, Phillip and Clematide, Simon",
  title = "Improving OCR of Black Letter in Historical Newspapers: The Unreasonable Effectiveness of HTR Models on Low-Resolution Images",
  year = 2019,
  booktitle = "Proceedings of the Digital Humanities 2019, (DH2019)"
}
```

Also cite Ströbel, Clematide & Volk (2020) as referenced in the project proposal. Add the full reference here once confirmed.
