from src.skinCancer import SCprediction
from src.yolo_object_detect import YOLO_Op
from src.maskNonMask import MNMRun


def run():
    while(True):

        try:
            print("""
            Enter only the index number:
            ============================

            1. SkinCancer
            2. Mask Non Mask
            3. Yolo Localizer
            4. Exit

            
            
            """)

            inp = int(input("Enter here : "))

            if(inp == 1):
                SCprediction()

            if(inp == 2):
                MNMRun()

            if(inp == 3):
                YOLO_Op()

            if (inp == 4):
                break
        except Exception as e:
            print(f"[INFO] Oops! An Error occurred restarting...")
            print(f"[ERROR] {e}")


if __name__ == "__main__":
    run()
