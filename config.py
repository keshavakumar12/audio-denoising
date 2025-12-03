import os

class TrainingConfig:
    CLEAN_AUDIO_DIR = r"D:\audio denoising\Data\Clean"
    NOISE_AUDIO_DIR = 'data/noise'
    MODELS_DIR = 'models'
    
    MODEL_TYPE = 'spectral_unet'
    
    NUM_TRAINING_SAMPLES = 2000
    NUM_VALIDATION_SAMPLES = 400
    SEGMENT_LENGTH = 3.0
    SAMPLE_RATE = 16000
    
    EPOCHS = 50
    BATCH_SIZE = 8
    LEARNING_RATE = 0.001
    
    SNR_RANGE = (0, 20)
    NOISE_TYPES = ['gaussian', 'colored', 'real']
    
    N_FFT = 1024
    HOP_LENGTH = 256
    
    EARLY_STOPPING_PATIENCE = 10
    REDUCE_LR_PATIENCE = 5
    REDUCE_LR_FACTOR = 0.5
    MIN_LEARNING_RATE = 1e-6
    
    VALIDATION_SPLIT = 0.2
    SHUFFLE_DATA = True
    RANDOM_SEED = 42
    
    SAVE_BEST_ONLY = True
    SAVE_TRAINING_PLOTS = True
    LOG_LEVEL = 'INFO'
    
    USE_GPU = True
    MIXED_PRECISION = False
    
    @classmethod
    def get_model_specific_config(cls, model_type):
        configs = {
            'spectral_unet': {
                'batch_size': 8,
                'epochs': 50,
                'learning_rate': 0.001,
                'loss': 'binary_crossentropy'
            },
            'rnn_denoiser': {
                'batch_size': 4,
                'epochs': 60,
                'learning_rate': 0.0005,
                'loss': 'binary_crossentropy'
            },
            'wavenet': {
                'batch_size': 2,
                'epochs': 80,
                'learning_rate': 0.0001,
                'loss': 'mse'
            }
        }
        return configs.get(model_type, configs['spectral_unet'])
    
    @classmethod
    def create_directories(cls):
        directories = [
            cls.CLEAN_AUDIO_DIR,
            cls.NOISE_AUDIO_DIR,
            cls.MODELS_DIR,
            'logs',
            'plots'
        ]
        
        for directory in directories:
            if directory:
                os.makedirs(directory, exist_ok=True)
    
    @classmethod
    def validate_config(cls):
        errors = []
        
        if not os.path.exists(cls.CLEAN_AUDIO_DIR):
            errors.append(f"Clean audio directory not found: {cls.CLEAN_AUDIO_DIR}")
        
        if cls.NOISE_AUDIO_DIR and not os.path.exists(cls.NOISE_AUDIO_DIR):
            errors.append(f"Noise audio directory not found: {cls.NOISE_AUDIO_DIR}")
        
        if cls.NUM_TRAINING_SAMPLES < 100:
            errors.append("NUM_TRAINING_SAMPLES should be at least 100")
        
        if cls.SEGMENT_LENGTH < 1.0:
            errors.append("SEGMENT_LENGTH should be at least 1.0 seconds")
        
        if cls.BATCH_SIZE < 1:
            errors.append("BATCH_SIZE must be positive")
        
        if cls.LEARNING_RATE <= 0:
            errors.append("LEARNING_RATE must be positive")
        
        valid_models = ['spectral_unet', 'rnn_denoiser', 'wavenet']
        if cls.MODEL_TYPE not in valid_models:
            errors.append(f"MODEL_TYPE must be one of: {valid_models}")
        
        return errors

class QuickStartConfig(TrainingConfig):
    NUM_TRAINING_SAMPLES = 200
    NUM_VALIDATION_SAMPLES = 50
    EPOCHS = 10
    BATCH_SIZE = 4

class ProductionConfig(TrainingConfig):
    NUM_TRAINING_SAMPLES = 10000
    NUM_VALIDATION_SAMPLES = 2000
    EPOCHS = 100
    BATCH_SIZE = 16
    EARLY_STOPPING_PATIENCE = 15
    MIXED_PRECISION = True

CONFIGS = {
    'quick': QuickStartConfig,
    'default': TrainingConfig,
    'production': ProductionConfig
}
