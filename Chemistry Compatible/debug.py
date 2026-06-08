import os
import sys
import traceback

print("="*60)
print("  TYSSUEFLOW2D MONITOR ACTIVATED - HOLDING TERMINAL...")
print("="*60)

try:
    # Force the absolute project path directory into python's runtime memory
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    print(f"[DEBUG INFO] Current Working Directory: {os.getcwd()}")
    print(f"[DEBUG INFO] Project Root Added: {PROJECT_ROOT}\n")

    # Manually trigger the framework import sequence to catch the exact broken file
    print("[DEBUG INFO] Checking framework integrity...")
    import main
    
    print("[DEBUG INFO] Booting Tkinter Event Loop Window...")
    # Launch main app
    import tkinter as tk
    root = tk.Tk()
    app = main.MainDashboardApp(root)
    root.mainloop()

except Exception as e:
    print("\n" + "!"*60)
    print("  [FATAL BOOT ENGINE CRASH CAPTURED BY LOGGER]")
    print("!"*60)
    print(f"\nError Message: {str(e)}")
    print("\n------------------ FULL SYSTEM TRACEBACK ------------------")
    traceback.print_exc()
    print("-----------------------------------------------------------\n")

finally:
    print("="*60)
    print("  TERMINAL LOCKED SAFE. WILL NOT CLOSE.")
    print("="*60)
    input("\nPress Enter key to close this window...")