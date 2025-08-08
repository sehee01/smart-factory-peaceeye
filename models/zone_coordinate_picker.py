import cv2
import json
import os
import argparse
import numpy as np
from datetime import datetime

class ZoneCoordinatePicker:
    def __init__(self, video_path):
        self.video_path = video_path
        self.coordinates = []
        self.current_frame = None
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.zone_name = "restricted_zone"
        self.threshold = 100  # 1 meter (100 pixels) warning radius
        
        # 마우스 콜백 설정
        cv2.namedWindow('Zone Coordinate Picker', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Zone Coordinate Picker', self.mouse_callback)
        
    def mouse_callback(self, event, x, y, flags, param):
        """Mouse event handling"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # First point
            self.start_point = (x, y)
            self.drawing = True
            print(f"[INFO] First point set: ({x}, {y})")
            
        elif event == cv2.EVENT_LBUTTONUP:
            # Second point
            self.end_point = (x, y)
            self.drawing = False
            
            # Coordinate alignment (x1,y1 is top-left, x2,y2 is bottom-right)
            x1, y1 = min(self.start_point[0], self.end_point[0]), min(self.start_point[1], self.end_point[1])
            x2, y2 = max(self.start_point[0], self.end_point[0]), max(self.start_point[1], self.end_point[1])
            
            # Save restricted zone information
            zone_info = {
                'x1': int(x1),
                'y1': int(y1),
                'x2': int(x2),
                'y2': int(y2),
                'threshold': self.threshold,
                'zone_name': self.zone_name,
                'created_at': datetime.now().isoformat()
            }
            
            self.coordinates.append(zone_info)
            print(f"[INFO] Restricted zone setup complete: ({x1},{y1}) ~ ({x2},{y2})")
            print(f"[INFO] Detection radius: {self.threshold} pixels")
            
            # Redraw frame
            self.draw_zones()
            
    def draw_zones(self):
        """Draw configured restricted zones on frame"""
        if self.current_frame is None:
            return
            
        # Copy frame
        display_frame = self.current_frame.copy()
        
        # Draw configured restricted zones
        for i, zone in enumerate(self.coordinates):
            x1, y1, x2, y2 = zone['x1'], zone['y1'], zone['x2'], zone['y2']
            
            # Draw rectangle
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Draw detection radius circle (center point based)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(display_frame, (center_x, center_y), zone['threshold'], (255, 0, 0), 2)
            
            # Display text
            cv2.putText(display_frame, f"Zone {i+1}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(display_frame, f"Threshold: {zone['threshold']}", (x1, y2+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Display currently drawing rectangle
        if self.drawing and self.start_point:
            cv2.rectangle(display_frame, self.start_point, 
                         (self.start_point[0], self.start_point[1]), (0, 255, 0), 2)
        
        # Guide text
        cv2.putText(display_frame, "Draw restricted zone with mouse (drag)", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, "R: Reset, S: Save, Q: Quit", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Zones set: {len(self.coordinates)}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Zone Coordinate Picker', display_frame)
    
    def save_coordinates(self, output_file):
        """Save coordinates to JSON file"""
        if not self.coordinates:
            print("[WARNING] No coordinates to save.")
            return False
            
        # Data structure to save
        save_data = {
            'video_path': self.video_path,
            'created_at': datetime.now().isoformat(),
            'zones': self.coordinates,
            'total_zones': len(self.coordinates)
        }
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            print(f"[SUCCESS] Coordinates saved to {output_file}")
            return True
        except Exception as e:
            print(f"[ERROR] File save failed: {e}")
            return False
    
    def load_coordinates(self, input_file):
        """Load coordinates from existing JSON file"""
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.coordinates = data.get('zones', [])
                print(f"[INFO] Loaded {len(self.coordinates)} restricted zones.")
                return True
        except Exception as e:
            print(f"[ERROR] File load failed: {e}")
            return False
    
    def run(self):
        """Main execution function"""
        # Open video capture
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video file: {self.video_path}")
            return
        
        # Get first frame
        ret, self.current_frame = cap.read()
        if not ret:
            print("[ERROR] Cannot read frame from video.")
            cap.release()
            return
        
        print("\n=== Restricted Zone Coordinate Setup Tool ===")
        print("Usage:")
        print("1. Drag mouse to draw restricted zone")
        print("2. R: Reset all coordinates")
        print("3. S: Save coordinates to JSON file")
        print("4. Q: Quit")
        print("=" * 50)
        
        # Display first frame
        self.draw_zones()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Q or ESC
                break
            elif key == ord('r'):  # R: Reset
                self.coordinates = []
                print("[INFO] All coordinates reset.")
                self.draw_zones()
            elif key == ord('s'):  # S: Save
                output_file = f"zone_coordinates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                self.save_coordinates(output_file)
            elif key == ord('l'):  # L: Load
                input_file = input("Enter JSON filename to load: ").strip()
                if input_file and os.path.exists(input_file):
                    self.load_coordinates(input_file)
                    self.draw_zones()
                else:
                    print("[ERROR] File not found.")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Auto-save confirmation on exit
        if self.coordinates:
            save_choice = input("\nAuto-save coordinates? (y/n): ").strip().lower()
            if save_choice == 'y':
                output_file = f"zone_coordinates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                self.save_coordinates(output_file)

def create_zone_config_from_json(json_file, output_config_file="restricted_zone_config.py"):
    """Generate restricted zone config file from JSON"""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        zones = data.get('zones', [])
        if not zones:
            print("[ERROR] No restricted zone data in JSON file.")
            return False
        
        # Generate config file content
        config_content = f'''# Restricted Zone Config File (Auto-generated)
# Created: {datetime.now().isoformat()}
# Source Video: {data.get('video_path', 'Unknown')}

# Alarm threshold settings
RESTRICTED_ZONE_CONFIG = {{
    'severity': 'critical',
    'alarm_threshold': 0,  # Critical alarm when entering restricted zone
    'warning_threshold': 100,  # Warning alarm within 1 meter
    'alarm_message': 'Restricted zone entry! Exit immediately!',
    'warning_message': 'Within 1 meter of restricted zone! Be careful!',
    'color': 'red'
}}

# Restricted zone coordinate settings (Auto-generated from JSON)
RESTRICTED_ZONE_EXAMPLES = {{
'''
        
        # Add each restricted zone to config
        for i, zone in enumerate(zones):
            zone_name = zone.get('zone_name', f'zone_{i+1}')
            config_content += f'''    '{zone_name}': {{
        'x1': {zone['x1']}, 'y1': {zone['y1']},
        'x2': {zone['x2']}, 'y2': {zone['y2']},
        'threshold': {zone['threshold']}
    }},
'''
        
        config_content += '''}

# Current restricted zone type setting
CURRENT_ZONE_TYPE = 'zone_1'

def get_zone_config():
    """Return current restricted zone configuration"""
    return RESTRICTED_ZONE_CONFIG

def get_zone_coordinates():
    """Return current restricted zone coordinates"""
    return RESTRICTED_ZONE_EXAMPLES.get(CURRENT_ZONE_TYPE, RESTRICTED_ZONE_EXAMPLES['zone_1'])

def set_zone_type(zone_type):
    """Set restricted zone type"""
    global CURRENT_ZONE_TYPE
    if zone_type in RESTRICTED_ZONE_EXAMPLES:
        CURRENT_ZONE_TYPE = zone_type
        print(f"[CONFIG] Restricted zone type set to '{{zone_type}}'.")
        return True
    else:
        print(f"[ERROR] Unknown restricted zone type: {{zone_type}}")
        print(f"[INFO] Available types: {{list(RESTRICTED_ZONE_EXAMPLES.keys())}}")
        return False

def add_custom_zone(zone_name, x1, y1, x2, y2, threshold):
    """Add custom restricted zone"""
    RESTRICTED_ZONE_EXAMPLES[zone_name] = {{
        'x1': x1, 'y1': y1,
        'x2': x2, 'y2': y2,
        'threshold': threshold
    }}
    print(f"[CONFIG] Custom restricted zone '{{zone_name}}' added.")
    print(f"[CONFIG] Coordinates: ({{x1}},{{y1}}) ~ ({{x2}},{{y2}}), Detection radius: {{threshold}}")

def list_available_zones():
    """Display available restricted zones"""
    print("\\n=== Available Restricted Zones ===")
    for zone_name, coords in RESTRICTED_ZONE_EXAMPLES.items():
        print(f"  {{zone_name}}: ({{coords['x1']}},{{coords['y1']}}) ~ ({{coords['x2']}},{{coords['y2']}}) [Radius: {{coords['threshold']}}]")
    print(f"\\nCurrent setting: {{CURRENT_ZONE_TYPE}}")
    print("=" * 40)

if __name__ == "__main__":
    # Config file test
    list_available_zones()
'''
        
        # Save config file
        with open(output_config_file, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        print(f"[SUCCESS] Restricted zone config file created: {output_config_file}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Config file creation failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Restricted Zone Coordinate Setup Tool')
    parser.add_argument('video_path', nargs='?', default="../test_video/KSEB03.mp4", 
                       help='Video file path (default: ../test_video/KSEB03.mp4)')
    parser.add_argument('--load', help='Load coordinates from existing JSON file')
    parser.add_argument('--create-config', help='Generate config file from JSON file')
    parser.add_argument('--list-videos', action='store_true', help='Display available video files')
    
    args = parser.parse_args()
    
    if args.create_config:
        # Generate config file from JSON
        create_zone_config_from_json(args.create_config)
        return
    
    if args.list_videos:
        # Display available video files
        print("=== Available Video Files ===")
        test_video_dir = "../test_video"
        if os.path.exists(test_video_dir):
            video_files = [f for f in os.listdir(test_video_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
            for i, video in enumerate(video_files, 1):
                print(f"{i}. {video}")
            print("=" * 30)
        else:
            print(f"[WARNING] Directory not found: {test_video_dir}")
        return
    
    # Check if video file exists
    if not os.path.exists(args.video_path):
        print(f"[ERROR] Video file not found: {args.video_path}")
        print(f"[INFO] Default path: {os.path.abspath(args.video_path)}")
        print("[INFO] Use --list-videos option to check available video files.")
        return
    
    print(f"[INFO] Loading video file: {args.video_path}")
    
    # Run coordinate setup tool
    picker = ZoneCoordinatePicker(args.video_path)
    
    # Load existing JSON file
    if args.load and os.path.exists(args.load):
        picker.load_coordinates(args.load)
    
    # Main execution
    picker.run()

if __name__ == "__main__":
    main()
