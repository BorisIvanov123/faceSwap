"""
Main Pipeline for Children's Book Generation with IP-Adapter FaceID

Usage:
    # Single page
    python pipeline.py --photo photos/faces/child.jpg --template photos/templates/page01.jpg --output photos/output_pages/page01.png --scene "playing in forest"

    # Whole book
    python pipeline.py --book config.json

    # Force use IP-Adapter (if available)
    python pipeline.py --photo ... --template ... --output ... --use-ipadapter
"""

import sys
from pathlib import Path

# Add core and utils to Python path
sys.path.insert(0, str(Path(__file__).parent / "core"))
sys.path.insert(0, str(Path(__file__).parent / "utils"))

from core.mask_generator import MaskGenerator

# Try to import both generators
try:
    from page_generator_ipadapter import IPAdapterBookGenerator
    HAS_IP_ADAPTER = True
except ImportError:
    HAS_IP_ADAPTER = False
    print("⚠️  IP-Adapter not available, using standard generator")

from core.page_generator import SimplifiedBookGenerator

import argparse
import json


class BookPipeline:
    """Main pipeline - coordinates everything"""

    def __init__(self, use_ipadapter=False):
        print("🚀 Initializing Pipeline...")

        self.mask_generator = MaskGenerator()

        # Choose generator based on availability and preference
        if use_ipadapter and HAS_IP_ADAPTER:
            print("   Using IP-Adapter FaceID generator (best quality)")
            self.page_generator = IPAdapterBookGenerator()
            self.using_ipadapter = True
        else:
            if use_ipadapter:
                print("   ⚠️  IP-Adapter requested but not available")
            print("   Using standard ControlNet generator")
            self.page_generator = SimplifiedBookGenerator()
            self.using_ipadapter = False

        print("✅ Ready!\n")

    def generate_single_page(
        self,
        child_photo: str,
        template: str,
        output: str,
        scene: str = "in a magical scene",
        force_remask: bool = False
    ):
        """Generate one page"""

        print(f"\n📖 Generating Page")
        print(f"Photo: {child_photo}")
        print(f"Template: {template}")
        print(f"Scene: {scene}")
        print(f"Generator: {'IP-Adapter FaceID' if self.using_ipadapter else 'ControlNet'}\n")

        # Step 1: Mask
        mask_path = Path("photos/masks") / f"{Path(template).stem}_mask.png"

        if not mask_path.exists() or force_remask:
            print("📝 Creating mask...")
            self.mask_generator.generate_smart_mask(
                template_path=template,
                mask_path=str(mask_path),
                method="ellipse"
            )
        else:
            print(f"📝 Using existing mask: {mask_path}")

        # Step 2: Generate
        print("\n🎨 Generating page...")
        self.page_generator.generate_page(
            child_photo_path=child_photo,
            template_path=template,
            mask_path=str(mask_path),
            output_path=output,
            scene_description=scene
        )

        print(f"\n✅ Done! View: open {output}\n")

    def generate_book(self, config_file: str):
        """Generate multiple pages from config"""

        with open(config_file) as f:
            config = json.load(f)

        title = config.get("title", "Untitled")
        child_photo = config.get("child_photo")
        pages = config.get("pages", [])

        print(f"\n📚 Generating Book: {title}")
        print(f"Pages: {len(pages)}")
        print(f"Generator: {'IP-Adapter FaceID' if self.using_ipadapter else 'ControlNet'}\n")

        for i, page in enumerate(pages, 1):
            print(f"\n{'='*60}")
            print(f"Page {i}/{len(pages)}")
            print(f"{'='*60}")

            self.generate_single_page(
                child_photo=child_photo,
                template=page["template"],
                output=page["output"],
                scene=page.get("scene", "")
            )

        print(f"\n{'='*60}")
        print(f"✅ Book Complete: {title}")
        print(f"   Generated: {len(pages)} pages")
        print(f"{'='*60}\n")

    def visualize_mask(self, template: str):
        """Visualize what will be replaced"""
        template_path = Path(template)
        mask_path = Path("photos/masks") / f"{template_path.stem}_mask.png"
        viz_path = Path("photos/masks") / f"{template_path.stem}_visualization.png"

        if not mask_path.exists():
            print("Generating mask first...")
            self.mask_generator.generate_smart_mask(
                template_path=str(template_path),
                mask_path=str(mask_path),
                method="ellipse"
            )

        self.mask_generator.visualize_mask(
            template_path=str(template_path),
            mask_path=str(mask_path),
            output_path=str(viz_path)
        )

        print(f"\n✅ Visualization saved to: {viz_path}")
        print(f"   Red area = what will be replaced")
        print(f"   View: open {viz_path}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate personalized book pages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single page with standard generator
  python pipeline.py --photo photos/faces/child.jpg --template photos/templates/page01.jpg --output photos/output_pages/page01.png --scene "playing in forest"
  
  # Single page with IP-Adapter (better face matching)
  python pipeline.py --photo photos/faces/child.jpg --template photos/templates/page01.jpg --output photos/output_pages/page01.png --scene "playing in forest" --use-ipadapter
  
  # Visualize mask before generation
  python pipeline.py --visualize-mask photos/templates/page01.jpg
  
  # Generate entire book
  python pipeline.py --book my_book_config.json --use-ipadapter
        """
    )

    # Single page mode
    parser.add_argument("--photo", help="Path to child's photo")
    parser.add_argument("--template", help="Path to template illustration")
    parser.add_argument("--output", help="Output path for generated page")
    parser.add_argument("--scene", default="in a magical scene", help="Scene description")

    # Book mode
    parser.add_argument("--book", help="Path to book config JSON file")

    # Options
    parser.add_argument("--use-ipadapter", action="store_true",
                       help="Use IP-Adapter FaceID for better face matching (requires setup)")
    parser.add_argument("--remask", action="store_true",
                       help="Force regenerate mask even if exists")
    parser.add_argument("--visualize-mask", metavar="TEMPLATE",
                       help="Visualize mask for a template (doesn't generate page)")

    args = parser.parse_args()

    # Handle visualize mode
    if args.visualize_mask:
        pipeline = BookPipeline(use_ipadapter=False)
        pipeline.visualize_mask(args.visualize_mask)
        return

    # Initialize pipeline
    pipeline = BookPipeline(use_ipadapter=args.use_ipadapter)

    # Run generation
    if args.book:
        # Book mode - multiple pages
        pipeline.generate_book(args.book)
    elif args.photo and args.template and args.output:
        # Single page mode
        pipeline.generate_single_page(
            child_photo=args.photo,
            template=args.template,
            output=args.output,
            scene=args.scene,
            force_remask=args.remask
        )
    else:
        print("❌ Error: Must provide either:")
        print("   Single page: --photo, --template, --output")
        print("   Book: --book config.json")
        print("   Visualize: --visualize-mask template.jpg")
        print("\nRun 'python pipeline.py --help' for more info")
        parser.print_help()


if __name__ == "__main__":
    main()