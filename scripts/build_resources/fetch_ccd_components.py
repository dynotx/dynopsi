"""Fetch CCD components for just AAs. Note that the molecular structures do have H's removed but are not optimized with force fields."""

import logging
import tempfile
import time
import urllib.request
from pathlib import Path

from pdbeccdutils.core import ccd_reader, ccd_writer

from dynopsi.data.constants import STANDARD_RESIDUES
from dynopsi.data.paths import AA_CCD_CIF_FILE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():

    # make sure the output path exists
    aa_ccd_cif_path = Path(AA_CCD_CIF_FILE)
    if not aa_ccd_cif_path.exists():
        aa_ccd_cif_path.parent.mkdir(parents=True, exist_ok=True)

    prev_time = time.time()
    logger.info("Fetching CCD components for just AAs")
    with tempfile.TemporaryDirectory() as tmpdir:
        components_cif_file = Path(tmpdir) / "components.cif"
        # PDB download
        FULL_CCD_CIF_URL = "https://files.wwpdb.org/pub/pdb/data/monomers/components.cif"

        def progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            pct = downloaded / total_size * 100
            print(f"\r{pct:.1f}% ({downloaded / 1e6:.1f} / {total_size / 1e6:.1f} MB)", end="")

        logger.info(f"Downloading CCD components from {FULL_CCD_CIF_URL} to {components_cif_file}")
        urllib.request.urlretrieve(FULL_CCD_CIF_URL, components_cif_file, reporthook=progress)
        logger.info(f"Downloaded CCD components in {time.time() - prev_time:.2f} seconds")
        prev_time = time.time()

        logger.info(f"Reading CCD components from {components_cif_file}")
        components = ccd_reader.read_pdb_components_file(str(components_cif_file))
        logger.info(f"Read CCD components in {time.time() - prev_time:.2f} seconds")
        prev_time = time.time()

        # make a subdir of just the AAs
        aa_subdir = Path(tmpdir) / "aa"
        aa_subdir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Filtering and writing CCD components for AAs to {AA_CCD_CIF_FILE}")
        for resname in [x for x in STANDARD_RESIDUES if x != "UNK"]:
            if resname not in components:
                logger.warning(f"Residue {resname} not found in components")
            else:
                component = components[resname]

                outfile = Path(aa_subdir) / f"{component.component.id}.cif"
                # this save automatically removes Hs
                ccd_writer.to_pdb_ccd_cif_file(path=str(outfile), component=component.component)

        output = ""
        for cif_file in aa_subdir.glob("*.cif"):
            output += cif_file.read_text()

    with open(aa_ccd_cif_path, "w") as f:
        f.write(output)

    logger.info(
        f"Filtered and wrote CCD components for AAs to {aa_ccd_cif_path} in {time.time() - prev_time:.2f} seconds"
    )


if __name__ == "__main__":
    main()
