from src.skinCancer import SCprediction
from src.yolo_object_detect import YOLO_Op
from src.maskNonMask import MaskedFacesClassifier


def run():
    while(True):

        try:
            print("""
            Enter only the index number:
            ============================

            1. SkinCancer
            2. Mask Non Mask
            3. Yolo Localizer
            5. Exit

            
            
            """)

            inp = int(input("Enter here : "))

            if(inp == 1):
                SCprediction()

            if(inp == 2):
                mask_detect = MaskedFacesClassifier()
                mask_detect.images()
                mask_detect.generate_video()

            if(inp == 3):
                YOLO_Op()

            if (inp == 5):
                break
        except Exception as e:
            print(f"[INFO] Oops! An Error occurred restarting...")
            print(f"[ERROR] {e}")


if __name__ == "__main__":
    run()
