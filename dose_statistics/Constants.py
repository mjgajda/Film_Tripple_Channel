import os
import numpy as np
import json

from dose_statistics.fileManipulation import extract_NIST_files, extract_snapshots_from_directories
def constants(postParentDir, preParentDir, parentDir, doseCurveJson):
    postParentDir = postParentDir
    preParentDir = preParentDir
    parentDir = parentDir
    doseCurveJson = doseCurveJson

    snapshotsPre = extract_snapshots_from_directories( [preParentDir])

    snapshotsPost = extract_snapshots_from_directories([postParentDir])

    NISTPre = extract_NIST_files(preParentDir)

    NISTPost = extract_NIST_files(postParentDir)

    doseCurveJson = doseCurveJson

    saveDirectory = os.path.join(parentDir, "dose_statistics_results")


    return postParentDir, snapshotsPost, preParentDir, snapshotsPre, NISTPost, NISTPre, doseCurveJson, saveDirectory

