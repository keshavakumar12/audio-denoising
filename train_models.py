import os
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir="
import numpy as np
import tensorflow as tf
try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import librosa
import soundfile as sf
import glob
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import logging
from datetime import datetime

from app import RNNDenoiser, WaveNetDenoiser, AudioProcessor
tf.config.optimizer.set_jit(False)
print("TF version:", tf.__version__)
print("Available devices:", tf.config.list_physical_devices())

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Using GPU devices:", gpus)
    except Exception as e:
        print("Could not set memory growth:", e)
else:
    print("No GPU visible to TensorFlow, training will use CPU.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleSpectralUNet:
    def __init__(self, input_shape=(128, 513)):
        self.input_shape = input_shape
        self.model = self._build_model()
    
    def _build_model(self):
        inputs = layers.Input(shape=self.input_shape)
        
        c1 = layers.Conv1D(64, 3, activation='relu', padding='same')(inputs)
        c1 = layers.Conv1D(64, 3, activation='relu', padding='same')(c1)
        p1 = layers.MaxPooling1D(2)(c1)
        
        c2 = layers.Conv1D(128, 3, activation='relu', padding='same')(p1)
        c2 = layers.Conv1D(128, 3, activation='relu', padding='same')(c2)
        p2 = layers.MaxPooling1D(2)(c2)
        
        c3 = layers.Conv1D(256, 3, activation='relu', padding='same')(p2)
        c3 = layers.Conv1D(256, 3, activation='relu', padding='same')(c3)
        p3 = layers.MaxPooling1D(2)(c3)
        
        c4 = layers.Conv1D(512, 3, activation='relu', padding='same')(p3)
        c4 = layers.Conv1D(512, 3, activation='relu', padding='same')(c4)
        p4 = layers.MaxPooling1D(2)(c4)
        
        c5 = layers.Conv1D(1024, 3, activation='relu', padding='same')(p4)
        c5 = layers.Conv1D(1024, 3, activation='relu', padding='same')(c5)
        
        u6 = layers.UpSampling1D(2)(c5)
        u6 = layers.Conv1D(512, 2, activation='relu', padding='same')(u6)
        m6 = layers.Concatenate()([c4, u6])
        c6 = layers.Conv1D(512, 3, activation='relu', padding='same')(m6)
        c6 = layers.Conv1D(512, 3, activation='relu', padding='same')(c6)
        
        u7 = layers.UpSampling1D(2)(c6)
        u7 = layers.Conv1D(256, 2, activation='relu', padding='same')(u7)
        m7 = layers.Concatenate()([c3, u7])
        c7 = layers.Conv1D(256, 3, activation='relu', padding='same')(m7)
        c7 = layers.Conv1D(256, 3, activation='relu', padding='same')(c7)
        
        u8 = layers.UpSampling1D(2)(c7)
        u8 = layers.Conv1D(128, 2, activation='relu', padding='same')(u8)
        m8 = layers.Concatenate()([c2, u8])
        c8 = layers.Conv1D(128, 3, activation='relu', padding='same')(m8)
        c8 = layers.Conv1D(128, 3, activation='relu', padding='same')(c8)
        
        u9 = layers.UpSampling1D(2)(c8)
        u9 = layers.Conv1D(64, 2, activation='relu', padding='same')(u9)
        m9 = layers.Concatenate()([c1, u9])
        c9 = layers.Conv1D(64, 3, activation='relu', padding='same')(m9)
        c9 = layers.Conv1D(64, 3, activation='relu', padding='same')(c9)
        
        outputs = layers.Conv1D(513, 1, activation='sigmoid')(c9)
        
        model = keras.Model(inputs=[inputs], outputs=[outputs])
        return model

class AudioDataGenerator:
    def __init__(self, clean_audio_dir, noise_audio_dir=None, sample_rate=16000):
        self.clean_audio_dir = clean_audio_dir
        self.noise_audio_dir = noise_audio_dir
        self.sample_rate = sample_rate
        self.audio_processor = AudioProcessor()
        self.clean_files = self._get_audio_files(clean_audio_dir)
        self.noise_files = self._get_audio_files(noise_audio_dir) if noise_audio_dir else []
        self.clean_files = self._filter_valid_audio_files(self.clean_files)
        if self.noise_files:
            self.noise_files = self._filter_valid_audio_files(self.noise_files)
        logger.info(f"Found {len(self.clean_files)} valid clean audio files")
        logger.info(f"Found {len(self.noise_files)} valid noise audio files")
        if len(self.clean_files) == 0:
            raise ValueError("No valid clean audio files found!")
    
    def _get_audio_files(self, directory):
        if not directory or not os.path.exists(directory):
            return []
        extensions = ['*.wav', '*.mp3', '*.flac', '*.m4a']
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(directory, '**', ext), recursive=True))
        return files
    
    def _filter_valid_audio_files(self, file_list, max_files=5000):
        valid_files = []
        files_to_check = file_list[:max_files] if len(file_list) > max_files else file_list
        logger.info(f"Validating {len(files_to_check)} audio files...")
        for i, file_path in enumerate(files_to_check):
            if i % 500 == 0 and i > 0:
                logger.info(f"Validated {i}/{len(files_to_check)} files, {len(valid_files)} valid so far")
            try:
                audio, sr = self.audio_processor.load_audio(file_path, self.sample_rate)
                if audio is not None and len(audio) > self.sample_rate:
                    valid_files.append(file_path)
                else:
                    logger.warning(f"Skipping file: {os.path.basename(file_path)}")
            except Exception:
                continue
        logger.info(f"Validation complete: {len(valid_files)}/{len(files_to_check)} valid")
        return valid_files

    def _safe_load_audio(self, file_path, max_retries=2):
        for attempt in range(max_retries):
            try:
                audio, sr = self.audio_processor.load_audio(file_path, self.sample_rate)
                if audio is not None and len(audio) > 0:
                    return audio, sr
                else:
                    return None, None
            except Exception:
                if attempt == max_retries - 1:
                    return None, None
        return None, None
    
    def add_synthetic_noise(self, clean_audio, noise_type='gaussian', snr_db=10):
        if noise_type == 'gaussian':
            noise_power = np.mean(clean_audio ** 2) / (10 ** (snr_db / 10))
            noise = np.random.normal(0, np.sqrt(noise_power), len(clean_audio))
        elif noise_type == 'colored':
            noise = np.random.normal(0, 1, len(clean_audio))
            noise = np.convolve(noise, [0.1, 0.2, 0.4, 0.2, 0.1], mode='same')
            noise_power = np.mean(clean_audio ** 2) / (10 ** (snr_db / 10))
            if np.mean(noise ** 2) > 0:
                noise = noise * np.sqrt(noise_power / np.mean(noise ** 2))
        elif noise_type == 'real' and self.noise_files:
            noise_file = random.choice(self.noise_files)
            noise_audio, _ = self._safe_load_audio(noise_file)
            if noise_audio is None:
                return self.add_synthetic_noise(clean_audio, 'gaussian', snr_db)
            if len(noise_audio) < len(clean_audio):
                repeats = int(np.ceil(len(clean_audio) / len(noise_audio)))
                noise_audio = np.tile(noise_audio, repeats)
            noise = noise_audio[:len(clean_audio)]
            signal_power = np.mean(clean_audio ** 2)
            noise_power = np.mean(noise ** 2)
            if noise_power > 0:
                desired_noise_power = signal_power / (10 ** (snr_db / 10))
                noise = noise * np.sqrt(desired_noise_power / noise_power)
            else:
                return self.add_synthetic_noise(clean_audio, 'gaussian', snr_db)
        else:
            noise = np.zeros_like(clean_audio)
        
        noisy_audio = clean_audio + noise
        return noisy_audio, noise
    
    def generate_training_pairs(self, num_samples=1000, segment_length=3.0):
        segment_samples = int(segment_length * self.sample_rate)
        X_noisy = []
        X_clean = []
        noise_types = ['gaussian', 'colored']
        if self.noise_files:
            noise_types.append('real')
        successful_samples = 0
        attempts = 0
        max_attempts = num_samples * 3
        while successful_samples < num_samples and attempts < max_attempts:
            attempts += 1
            clean_file = random.choice(self.clean_files)
            clean_audio, _ = self._safe_load_audio(clean_file)
            if clean_audio is None:
                continue
            if len(clean_audio) < segment_samples:
                continue
            try:
                start_idx = random.randint(0, len(clean_audio) - segment_samples)
                clean_segment = clean_audio[start_idx:start_idx + segment_samples]
                noise_type = random.choice(noise_types)
                snr_db = random.uniform(0, 20)
                noisy_segment, _ = self.add_synthetic_noise(clean_segment, noise_type, snr_db)
                if not np.isfinite(noisy_segment).all() or not np.isfinite(clean_segment).all():
                    continue
                X_noisy.append(noisy_segment)
                X_clean.append(clean_segment)
                successful_samples += 1
            except Exception:
                continue
        logger.info(f"Successfully generated {successful_samples} training pairs")
        return np.array(X_noisy), np.array(X_clean)

class FixedSizeSpectralDataPreprocessor:
    def __init__(self, n_fft=1024, hop_length=256, target_time_steps=128):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.target_time_steps = target_time_steps
        self.audio_processor = AudioProcessor()
    
    def _pad_or_crop_spectrogram(self, spectrogram):
        current_time_steps = spectrogram.shape[1]
        if current_time_steps < self.target_time_steps:
            pad_amount = self.target_time_steps - current_time_steps
            pad_left = pad_amount // 2
            pad_right = pad_amount - pad_left
            spectrogram = np.pad(spectrogram, ((0, 0), (pad_left, pad_right)), mode='constant')
        elif current_time_steps > self.target_time_steps:
            start_idx = (current_time_steps - self.target_time_steps) // 2
            spectrogram = spectrogram[:, start_idx:start_idx + self.target_time_steps]
        return spectrogram
    
    def prepare_training_data(self, noisy_audio, clean_audio):
        X_magnitude = []
        y_masks = []
        for i in range(len(noisy_audio)):
            try:
                noisy_mag, noisy_phase, _ = self.audio_processor.compute_stft(
                    noisy_audio[i], self.n_fft, self.hop_length
                )
                clean_mag, _, _ = self.audio_processor.compute_stft(
                    clean_audio[i], self.n_fft, self.hop_length
                )
                noisy_mag = self._pad_or_crop_spectrogram(noisy_mag)
                clean_mag = self._pad_or_crop_spectrogram(clean_mag)
                mask = clean_mag / (noisy_mag + 1e-8)
                mask = np.clip(mask, 0, 1)
                X_magnitude.append(noisy_mag.T)
                y_masks.append(mask.T)
            except Exception:
                continue
        return np.array(X_magnitude), np.array(y_masks)

class WaveformDataPreprocessor:
    def prepare_training_data(self, noisy_audio, clean_audio):
        X_noisy = np.expand_dims(noisy_audio, axis=-1)
        y_clean = np.expand_dims(clean_audio, axis=-1)
        return X_noisy, y_clean

class AudioDenoiserTrainer:
    def __init__(self, model_type='spectral_unet'):
        self.model_type = model_type
        self.model = None
        self.history = None
        if model_type in ['spectral_unet', 'rnn_denoiser']:
            self.preprocessor = FixedSizeSpectralDataPreprocessor(target_time_steps=128)
        else:
            self.preprocessor = WaveformDataPreprocessor()
    
    def create_model(self):
        if self.model_type == 'spectral_unet':
            model_wrapper = SimpleSpectralUNet(input_shape=(128, 513))
        elif self.model_type == 'rnn_denoiser':
            model_wrapper = RNNDenoiser()
        elif self.model_type == 'wavenet':
            model_wrapper = WaveNetDenoiser()
        else:
            raise ValueError(self.model_type)
        self.model = model_wrapper.model
        if self.model_type == 'wavenet':
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
        else:
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['mae']
            )
        logger.info(f"Created {self.model_type} model")
        self.model.summary()
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=8):
        if self.model is None:
            self.create_model()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_save_path = f'models/{self.model_type}_weights_{timestamp}.h5'
        callbacks_list = [
            callbacks.ModelCheckpoint(
                model_save_path,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            )
        ]
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=1
        )
        final_model_path = f'models/{self.model_type}_weights.h5'
        self.model.save_weights(final_model_path)
        logger.info(f"Model weights saved to {final_model_path}")
        return self.history
    
    def plot_training_history(self):
        if self.history is None:
            return
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title(f'{self.model_type} Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['mae'], label='Training MAE')
        plt.plot(self.history.history['val_mae'], label='Validation MAE')
        plt.title(f'{self.model_type} Training MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'training_history_{self.model_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.show()

from config import TrainingConfig

def main():
    CLEAN_AUDIO_DIR = TrainingConfig.CLEAN_AUDIO_DIR
    NOISE_AUDIO_DIR = TrainingConfig.NOISE_AUDIO_DIR
    MODEL_TYPE = 'spectral_unet'
    NUM_TRAINING_SAMPLES = 1000
    NUM_VALIDATION_SAMPLES = 200
    SEGMENT_LENGTH = 3.0
    EPOCHS = 30
    BATCH_SIZE = 16
    os.makedirs('models', exist_ok=True)
    os.makedirs('data/clean', exist_ok=True)
    os.makedirs('data/noise', exist_ok=True)
    logger.info("Starting audio denoiser training...")
    logger.info(f"Model type: {MODEL_TYPE}")
    if not os.path.exists(CLEAN_AUDIO_DIR) or not os.listdir(CLEAN_AUDIO_DIR):
        logger.error(f"Clean audio directory '{CLEAN_AUDIO_DIR}' is empty or doesn't exist!")
        return
    try:
        data_generator = AudioDataGenerator(CLEAN_AUDIO_DIR, NOISE_AUDIO_DIR)
        logger.info("Generating training data...")
        X_noisy, X_clean = data_generator.generate_training_pairs(
            NUM_TRAINING_SAMPLES + NUM_VALIDATION_SAMPLES, 
            SEGMENT_LENGTH
        )
        if len(X_noisy) == 0:
            logger.error("No training data!")
            return
        split_ratio = NUM_VALIDATION_SAMPLES / (NUM_TRAINING_SAMPLES + NUM_VALIDATION_SAMPLES)
        X_noisy_train, X_noisy_val, X_clean_train, X_clean_val = train_test_split(
            X_noisy, X_clean, test_size=split_ratio, random_state=42
        )
        logger.info(f"Training samples: {len(X_noisy_train)}")
        logger.info(f"Validation samples: {len(X_noisy_val)}")
        trainer = AudioDenoiserTrainer(MODEL_TYPE)
        logger.info("Preprocessing data...")
        X_train, y_train = trainer.preprocessor.prepare_training_data(X_noisy_train, X_clean_train)
        X_val, y_val = trainer.preprocessor.prepare_training_data(X_noisy_val, X_clean_val)
        if len(X_train) == 0 or len(X_val) == 0:
            logger.error("No data left after preprocessing!")
            return
        history = trainer.train(X_train, y_train, X_val, y_val, EPOCHS, BATCH_SIZE)
        trainer.plot_training_history()
        logger.info("Training completed!")
        logger.info(f"Model saved as models/{MODEL_TYPE}_weights.h5")
        if len(X_noisy_val) > 0:
            test_sample_idx = 0
            test_noisy = X_noisy_val[test_sample_idx:test_sample_idx+1]
            test_clean = X_clean_val[test_sample_idx:test_sample_idx+1]
            sf.write(f'test_noisy_{MODEL_TYPE}.wav', test_noisy[0], 16000)
            sf.write(f'test_clean_{MODEL_TYPE}.wav', test_clean[0], 16000)
            logger.info("Test files saved")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
