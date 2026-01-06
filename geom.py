""" Module editing geometry XML files for B0 trackers """

import os
import xml.etree.ElementTree as ET
import shutil
from pathlib import Path


def editGeom(x, b0_xml_job):
    """
    Update z-start positions of B0 tracker layers in a job-specific XML file.

    Args:
        x (tuple): (z1, dz2, dz3, dz4) geometry offsets.
        b0_xml_job (str): Path to B0_tracker_<jobid>.xml.

    Returns:
        str: Path to the modified XML file.
    """
    z1, dz2, dz3, dz4 = x

    custom1 = {"B0TrackerLayer1_zstart": f"B0Tracker_length/2.0+{z1}"}
    custom2 = {"B0TrackerLayer2_zstart": f"B0Tracker_length/2.0+{z1}+{dz2}"}
    custom3 = {"B0TrackerLayer3_zstart": f"B0Tracker_length/2.0+{z1}+{dz2}+{dz3}"}
    custom4 = {"B0TrackerLayer4_zstart": f"B0Tracker_length/2.0+{z1}+{dz2}+{dz3}+{dz4}"}

    tree = ET.parse(b0_xml_job)
    root = tree.getroot()
    found = set()

    for const in root.findall(".//constant"):
        name = const.get("name")
        if name in custom1:
            const.set("value", custom1[name])
            found.add(name)
        if name in custom2:
            const.set("value", custom2[name])
            found.add(name)
        if name in custom3:
            const.set("value", custom3[name])
            found.add(name)
        if name in custom4:
            const.set("value", custom4[name])
            found.add(name)

    tree.write(b0_xml_job, encoding="utf-8", xml_declaration=True)

    print("EPIC FILE UPDATED")

    return b0_xml_job


def editEPIC(epic_xml_job, default_old, default_new):
    """
    Replace default.xml include in epic_<jobid>.xml by a job-specific default file.

    Args:
        epic_xml_job (str): Path to epic_<jobid>.xml.
        default_old (str): Original default XML reference.
        default_new (str): Job-specific default XML reference.
    """
    default_old = default_old.replace("\\", "/")
    default_new = default_new.replace("\\", "/")
    old_name = os.path.basename(default_old)

    tree = ET.parse(epic_xml_job)
    root = tree.getroot()

    for element in root.findall(".//include"):
        ref = element.get("ref")
        print("[DEBUG geom.editEPIC] Found include ref in EPIC =", ref)
        if ref and ref.endswith(old_name):
            new_ref = ref.replace(old_name, os.path.basename(default_new))
            element.set("ref", new_ref)
            tree.write(epic_xml_job)
            print(f"[DEBUG geom.editEPIC] Updated EPIC include:\n  {ref}\n→ {new_ref}")
            return

    print("[WARN geom.editEPIC] failed to update EPIC XML include for far_forward/default.xml")


def editFarForwardDefault(default_xml_job, b0_old, b0_new):
    """
    Replace B0 tracker include in far_forward default XML by a job-specific file.

    Args:
        default_xml_job (str): Path to job-specific default XML.
        b0_old (str): Original B0 tracker XML reference.
        b0_new (str): Job-specific B0 tracker XML reference.
    """
    b0_old = b0_old.replace("\\", "/")
    b0_new = b0_new.replace("\\", "/")
    old_name = os.path.basename(b0_old)
    new_name = os.path.basename(b0_new)

    tree = ET.parse(default_xml_job)
    root = tree.getroot()

    for element in root.findall(".//include"):
        ref = element.get("ref")
        print("[DEBUG geom.editFarForwardDefault] Found include ref in far_forward/default =", ref)
        if ref and ref.endswith(old_name):
            new_ref = ref.replace(old_name, new_name)
            element.set("ref", new_ref)
            tree.write(default_xml_job)
            print(f"[DEBUG geom.editFarForwardDefault] Updated far_forward/default include:\n  {ref}\n→ {new_ref}")
            return

    print("[WARN DEBUG geom.editFarForwardDefault] failed to update far_forward default B0 include")


def create_dir(dir_name):
    """
    Create a directory if it does not exist.

    Args:
        dir_name (str): Directory path.
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def create_xml(x, jobid):
    """
    Create and modify a job-specific B0 tracker XML file.

    Args:
        x (tuple): (z1, dz2, dz3, dz4) geometry offsets.
        jobid (str | int): Job identifier.

    Returns:
        str: Path to B0_tracker_<jobid>.xml.
    """
    b0_xml = Path(os.environ["EIC_SOFTWARE"]) / "share/epic/compact/far_forward/B0_tracker.xml"
    b0_xml_job = (
        Path(os.environ["AIDE_WORKDIR"])
        / "share/epic/compact/far_forward"
        / f"B0_tracker_{jobid}.xml"
    )
    b0_xml_job.parent.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(b0_xml, b0_xml_job)

    # --- Debug: dump XML before/after geometry update ---
    print("===== CONTENT OF B0 TRACKER XML (BEFORE MODIF) =====")
    with open(b0_xml_job, "r") as f:
        print(f.read())
    print("=============================================================")

    editGeom(x, str(b0_xml_job))

    print("===== CONTENT OF B0 TRACKER XML (AFTER MODIF) =====")
    with open(b0_xml_job, "r") as f:
        print(f.read())
    print("=============================================================")

    return str(b0_xml_job)
