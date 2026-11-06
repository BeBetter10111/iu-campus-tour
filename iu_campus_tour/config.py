# Window
WINDOW_TITLE = "IU Campus Virtual Tour"
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
VSYNC = True            
MSAA_SAMPLES = 4        

# OpenGL
GL_MAJOR = 3
GL_MINOR = 3            

GL_ERROR_CHECKING = True

# Render
CLEAR_COLOR = (0.53, 0.81, 0.92, 1.0)   #
FPS_CAP = 0            
FPS_TITLE_INTERVAL = 0.5  

# Camera
FOV = 60.0            
NEAR_PLANE = 0.1
FAR_PLANE = 500.0
EYE_HEIGHT = 1.7      
WALK_SPEED = 4.0      # m/s
RUN_SPEED = 8.0       # m/s 
MOUSE_SENSITIVITY = 0.1  

#Đường dẫn assets 
from pathlib import Path
ASSET_DIR = Path(__file__).resolve().parent / "assets"
TEXTURE_DIR = ASSET_DIR / "textures"
MODEL_DIR = ASSET_DIR / "models"
SKYBOX_DIR = ASSET_DIR / "skybox"