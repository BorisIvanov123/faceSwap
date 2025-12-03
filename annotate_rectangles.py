import cv2
import json
import os

PAGES_DIR = "photos/templates"   # your 30 PNG pages
OUT_JSON = "templates/face_regions.json"

os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)

page_files = sorted([
    f for f in os.listdir(PAGES_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])

regions = {}

drawing = False
start = None
end = None
img = None
clone = None

def mouse_handler(event, x, y, flags, param):
    global drawing, start, end, img, clone

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start = (x, y)
        end = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        end = (x, y)
        clone = img.copy()
        cv2.rectangle(clone, start, end, (0,255,0), 2)
        cv2.imshow("Annotate", clone)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end = (x, y)
        clone = img.copy()
        cv2.rectangle(clone, start, end, (0,255,0), 2)
        cv2.imshow("Annotate", clone)


print("\n=== Annotation Tool Started ===")
print("Instructions:")
print(" - Click & drag a rectangle over the *child's face*")
print(" - Press ENTER to save rectangle for this page")
print(" - Press R to reset/redraw")
print(" - Press Q to quit early\n")

for page in page_files:
    img_path = os.path.join(PAGES_DIR, page)
    img = cv2.imread(img_path)
    clone = img.copy()

    print(f"\nAnnotating {page}")
    cv2.imshow("Annotate", clone)
    cv2.setMouseCallback("Annotate", mouse_handler)

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):  # reset
            clone = img.copy()
            cv2.imshow("Annotate", clone)
            start = None
            end = None
            print("🔄 Rectangle reset.")

        if key == ord('q'):  # quit
            print("❌ Early quit.")
            break

        if key == 13:  # ENTER
            if start and end:
                x1, y1 = start
                x2, y2 = end
                regions[page] = [int(x1), int(y1), int(x2), int(y2)]
                print("✔ Saved:", regions[page])
                break
            else:
                print("❌ Draw a rectangle first.")

    if key == ord('q'):
        break

cv2.destroyAllWindows()

# Save JSON
with open(OUT_JSON, "w") as f:
    json.dump(regions, f, indent=4)

print("\n🎉 Annotation complete!")
print("Saved:", OUT_JSON)
