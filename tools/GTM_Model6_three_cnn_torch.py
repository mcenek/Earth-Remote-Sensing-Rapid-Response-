"""Public entry point for the bounded native PyTorch synthetic three-CNN run."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"research"/"GTM_Model6"))
from GTM_Model6_three_cnn_torch import main

if __name__=="__main__":
    main()
