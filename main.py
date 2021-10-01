from src.skinCancer import SCprediction
from src.yolo_object_detect import YOLO_Op


def run():
    while(True):

        try:
            print("""
            Enter only the index number:
            ============================

            1. SkinCancer
            2. Mask Non Mask
            3. DrowsyDetection
            4. Yolo Localizer
            5. Exit

            
            
            """)

            inp = int(input("Enter here : "))

            if(inp == 1):
                SCprediction()

            if(inp == 4):
                YOLO_Op()

            if (inp == 5):
                break
        except:
            print("Oops! An Error occurred restarting...")


if __name__ == "__main__":
    run()
