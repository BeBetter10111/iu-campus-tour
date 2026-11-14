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

# Assets 
from pathlib import Path
ASSET_DIR = Path(__file__).resolve().parent / "assets"
TEXTURE_DIR = ASSET_DIR / "textures"
MODEL_DIR = ASSET_DIR / "models"
SKYBOX_DIR = ASSET_DIR / "skybox"

# Lighting
SUN_AZIMUTH = 160.0     
SUN_ELEVATION = 35.0   
SUN_COLOR = (1.0, 0.95, 0.85)
SUN_INTENSITY = 1.0
AMBIENT_COLOR = (0.25, 0.28, 0.35)   

# Shadow mapping
SHADOW_MAP_SIZE = 2048
SHADOW_RADIUS = 80.0        
SHADOW_DISTANCE = 150.0     
SHADOW_BIAS = 0.0002        
SHADOW_NORMAL_OFFSET = 1.0  