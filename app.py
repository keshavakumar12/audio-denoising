from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os
import io
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import librosa
import soundfile as sf
import logging
from werkzeug.utils import secure_filename
import tempfile
import uuid
from scipy import ndimage, signal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///audio_denoiser.db'
@app.route('/')
def index():
    # Serve index.html from project root folder
    return send_from_directory('.', 'index.html')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

db = SQLAlchemy(app)
CORS(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(100), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    total_processed = db.Column(db.Integer, default=0, nullable=False)
    total_duration_minutes = db.Column(db.Float, default=0.0, nullable=False)
    avg_noise_reduction_db = db.Column(db.Float, default=0.0, nullable=False)

class ProcessingLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(100), nullable=False)
    original_filename = db.Column(db.String(255))
    model_type = db.Column(db.String(50), nullable=False)
    noise_reduction_strength = db.Column(db.Float, nullable=False)
    processing_time = db.Column(db.Float)
    audio_duration = db.Column(db.Float)
    sample_rate = db.Column(db.Integer)
    noise_reduction_db = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class AudioProcessor:
    @staticmethod
    def load_audio(file_path, target_sr=16000):
        try:
            audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
            return audio, sr
        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            return None, None

    @staticmethod
    def compute_stft(audio, n_fft=1024, hop_length=256):
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        return np.abs(stft), np.angle(stft), stft

    @staticmethod
    def reconstruct_audio(magnitude, phase, hop_length=256):
        return librosa.istft(magnitude * np.exp(1j * phase), hop_length=hop_length)

    @staticmethod
    def normalize_audio(audio):
        max_val = np.max(np.abs(audio))
        return audio / max_val if max_val > 0 else audio

    @staticmethod
    def calculate_snr(clean_audio, noisy_audio):
        try:
            signal_power = np.mean(clean_audio ** 2)
            noise_power = np.mean((noisy_audio - clean_audio) ** 2)
            if noise_power < 1e-10:
                return 50
            snr = 10 * np.log10(signal_power / noise_power)
            return np.clip(snr, 0, 50)
        except:
            return 10

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
        return keras.Model(inputs=[inputs], outputs=[outputs])

class SimpleDenoiser:
    def __init__(self, input_shape=(None, 513)):
        self.input_shape = input_shape
        self.model = self._build_model()

    def _build_model(self):
        inputs = keras.Input(shape=self.input_shape)
        x = layers.Dense(256, activation='relu')(inputs)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(513, activation='sigmoid')(x)
        return keras.Model(inputs=inputs, outputs=outputs)


class RNNDenoiser:

    def __init__(self,
                 input_shape=(128, 513),
                 lstm_units=128,
                 num_layers=3,
                 dropout=0.2):
        self.input_shape = input_shape
        self.lstm_units = lstm_units
        self.num_layers = num_layers
        self.dropout = dropout
        self.model = self._build_model()

    def _build_model(self):
        inputs = layers.Input(shape=self.input_shape, name="noisy_mag_input")

      
        x = layers.LayerNormalization(axis=-1, name="input_layer_norm")(inputs)

     
        for i in range(self.num_layers):
            x = layers.LSTM(
                self.lstm_units,
                return_sequences=True,
                dropout=self.dropout,
                name=f"lstm_{i+1}"
            )(x)


        x = layers.TimeDistributed(
            layers.Dense(self.lstm_units, activation="relu"),
            name="td_dense_hidden"
        )(x)

        outputs = layers.TimeDistributed(
            layers.Dense(self.input_shape[1], activation="sigmoid"),
            name="td_mask_output"
        )(x)

        return keras.Model(
            inputs=inputs,
            outputs=outputs,
            name="rnn_lstm_spectral_denoiser"
        )


class WaveNetDenoiser:
    def __init__(self,
                 input_shape=(None, 1),
                 residual_channels=64,
                 skip_channels=64,
                 num_blocks=2,
                 dilations=(1, 2, 4, 8, 16, 32)):
        self.input_shape = input_shape
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.num_blocks = num_blocks
        self.dilations = dilations
        self.model = self._build_model()

    def _residual_block(self, x, dilation_rate, block_id, layer_id):
       
        name_prefix = f"b{block_id}_d{dilation_rate}_l{layer_id}"

        conv_filter = layers.Conv1D(
            self.residual_channels,
            kernel_size=3,
            dilation_rate=dilation_rate,
            padding="causal",
            name=f"{name_prefix}_conv_f"
        )(x)

        conv_gate = layers.Conv1D(
            self.residual_channels,
            kernel_size=3,
            dilation_rate=dilation_rate,
            padding="causal",
            name=f"{name_prefix}_conv_g"
        )(x)

        z = layers.Activation("tanh", name=f"{name_prefix}_tanh")(conv_filter)
        g = layers.Activation("sigmoid", name=f"{name_prefix}_sigmoid")(conv_gate)
        gated = layers.Multiply(name=f"{name_prefix}_gated")([z, g])

      
        residual = layers.Conv1D(
            self.residual_channels,
            kernel_size=1,
            padding="same",
            name=f"{name_prefix}_res_conv"
        )(gated)
        residual = layers.Add(name=f"{name_prefix}_res_add")([x, residual])

    
        skip = layers.Conv1D(
            self.skip_channels,
            kernel_size=1,
            padding="same",
            name=f"{name_prefix}_skip_conv"
        )(gated)

        return residual, skip

    def _build_model(self):
        inputs = layers.Input(shape=self.input_shape, name="noisy_waveform")

        x = layers.Conv1D(
            self.residual_channels,
            kernel_size=1,
            padding="causal",
            name="input_proj"
        )(inputs)

        skips = []


        for b in range(self.num_blocks):
            for i, d in enumerate(self.dilations):
                x, skip = self._residual_block(
                    x,
                    dilation_rate=d,
                    block_id=b,
                    layer_id=i
                )
                skips.append(skip)

        x = layers.Add(name="skip_add")(skips)
        x = layers.Activation("relu", name="skip_relu")(x)

        x = layers.Conv1D(
            self.skip_channels,
            kernel_size=1,
            activation="relu",
            padding="same",
            name="post_conv1"
        )(x)
        x = layers.Conv1D(
            self.skip_channels,
            kernel_size=1,
            activation="relu",
            padding="same",
            name="post_conv2"
        )(x)

        outputs = layers.Conv1D(
            1,
            kernel_size=1,
            activation="tanh",
            padding="same",
            name="waveform_out"
        )(x)

        return keras.Model(
            inputs=inputs,
            outputs=outputs,
            name="wavenet_denoiser"
        )


class AudioModelManager:
    def __init__(self):
        self.models = {}
        self.audio_processor = AudioProcessor()
        self.load_models()

    def load_models(self):
        try:
            
                try:
                    self.models['spectral_unet'] = SimpleSpectralUNet()
                    self.models['spectral_unet'].model.load_weights("models/spectral_unet_weights.h5")
                    logger.info("✅ Loaded SpectralUNet model with trained weights")
                except FileNotFoundError:
                    logger.warning("Trained SpectralUNet weights not found, using untrained model")
                    self.models['spectral_unet'] = SimpleSpectralUNet()
                except Exception as e:
                    logger.error(f"Failed to load SpectralUNet: {e}")
                    self.models['spectral_unet'] = SimpleDenoiser()
                    logger.info("Using SimpleDenoiser as fallback for SpectralUNet")

              
                try:
                    self.models['rnn_denoiser'] = RNNDenoiser()
                    self.models['rnn_denoiser'].model.load_weights("models/rnn_denoiser_weights.h5")
                    logger.info("✅ Loaded RNNDenoiser model with trained weights")
                except FileNotFoundError:
                    logger.warning("RNN weights not found, using untrained RNNDenoiser")
                    self.models['rnn_denoiser'] = RNNDenoiser()
                except Exception as e:
                    logger.error(f"Failed to load RNNDenoiser: {e}")
                    self.models['rnn_denoiser'] = SimpleDenoiser()
                    logger.info("Using SimpleDenoiser as fallback for RNN")

                
                try:
                    self.models['wavenet'] = WaveNetDenoiser()
                    self.models['wavenet'].model.load_weights("models/wavenet_weights.h5")
                    logger.info("✅ Loaded WaveNetDenoiser model with trained weights")
                except FileNotFoundError:
                    logger.warning("WaveNet weights not found, using untrained WaveNetDenoiser")
                    self.models['wavenet'] = WaveNetDenoiser()
                except Exception as e:
                    logger.error(f"Failed to load WaveNetDenoiser: {e}")
                    self.models['wavenet'] = SimpleDenoiser()
                    logger.info("Using SimpleDenoiser as fallback for WaveNet")

        
                for model_name, model_wrapper in self.models.items():
                    model_wrapper.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                    logger.info(f"Compiled {model_name} successfully")

        except Exception as e:
                logger.error(f"Error loading models: {e}")

    def denoise_audio(self, audio_path, model_type='spectral_unet', strength=0.8):
        try:
            model_wrapper = self.models.get(model_type, self.models['spectral_unet'])
            audio, sr = self.audio_processor.load_audio(audio_path)
            if audio is None:
                return None, None, 0
            original_audio = audio.copy()

            if model_type in ['spectral_unet', 'rnn_denoiser'] and hasattr(model_wrapper, 'model'):
                try:
                    logger.info("Using trained SpectralUNet model")
                    magnitude, phase, _ = self.audio_processor.compute_stft(audio)

                 
                    target_time_steps = 128
                    hop_steps = 64  

                    full_mask = self._predict_full_mask(
                        model_wrapper.model,
                        magnitude,
                        target_time_steps=target_time_steps,
                        hop_steps=hop_steps
                    )

                    denoised_magnitude = magnitude * full_mask
                    denoised_audio = self.audio_processor.reconstruct_audio(denoised_magnitude, phase)

              
                    if len(denoised_audio) > len(audio):
                        denoised_audio = denoised_audio[:len(audio)]
                    elif len(denoised_audio) < len(audio):
                        denoised_audio = np.pad(denoised_audio, (0, len(audio) - len(denoised_audio)), mode='constant')
                except Exception as spectral_error:
                    logger.warning(f"Spectral model processing failed: {spectral_error}. Using fallback.")
                    return self._fallback_processing(audio, strength, sr)
            elif model_type == 'wavenet' and hasattr(model_wrapper, 'model'):
                try:
                    logger.info("Using WaveNet model for time-domain denoising")
                    denoised_audio = self._wavenet_denoise_waveform(
                        model_wrapper.model,
                        audio,
                        sr,
                        segment_seconds=1.0,
                        hop_seconds=0.5
                    )   
                except Exception as wavenet_error:
                    logger.warning(f"WaveNet processing failed: {wavenet_error}. Using fallback.")
                    return self._fallback_processing(audio, strength, sr)

            else:
                return self._fallback_processing(audio, strength, sr)

            denoised_audio = self.audio_processor.normalize_audio(denoised_audio)
            noise_reduction_db = self.audio_processor.calculate_snr(denoised_audio, original_audio)
            return denoised_audio, sr, abs(noise_reduction_db)
        except Exception as e:
            logger.error(f"Error in audio denoising: {e}")
            return self._fallback_processing(audio, strength, sr)

    def _fallback_processing(self, audio, strength, sr):
        try:
            magnitude, phase, _ = self.audio_processor.compute_stft(audio)
            noise_floor = np.percentile(magnitude, 20, axis=1, keepdims=True)
            threshold_factor = 2.0 - strength
            mask = np.where(magnitude > noise_floor * threshold_factor, 1.0, 0.1 + 0.2 * strength)
            denoised_magnitude = magnitude * mask
            denoised_audio = self.audio_processor.reconstruct_audio(denoised_magnitude, phase)
            denoised_audio = self.audio_processor.normalize_audio(denoised_audio)
            noise_reduction_db = self.audio_processor.calculate_snr(denoised_audio, audio)
            return denoised_audio, sr, abs(noise_reduction_db)
        except:
            return audio * 0.9, sr, 0
    def _predict_full_mask(self, model, magnitude, target_time_steps=128, hop_steps=64):
       
        T = magnitude.shape[1]
        F = magnitude.shape[0]
        full_mask = np.zeros((F, T), dtype=np.float32)
        weight = np.zeros((T,), dtype=np.float32)

        # Smooth blending window over time dimension (Hann)
        win = np.hanning(target_time_steps).astype(np.float32)
        if target_time_steps == 1:
            win[:] = 1.0

        t = 0
        while t < T:
            end = t + target_time_steps
            if end <= T:
                chunk_mag = magnitude[:, t:end]
            else:
                pad = end - T
                chunk_mag = np.pad(magnitude[:, t:T], ((0, 0), (0, pad)), mode='constant')

            model_input = chunk_mag.T[np.newaxis, :, :].astype(np.float32)
            pred = model.predict(model_input, verbose=0)[0]  # (time, freq)
            chunk_mask = np.clip(pred.T, 0.0, 1.0)

        
            if chunk_mask.shape[1] == target_time_steps:
                w = win
            else:
                w = np.ones((chunk_mask.shape[1],), dtype=np.float32)

            if end <= T:
                full_mask[:, t:end] += chunk_mask * w[np.newaxis, :]
                weight[t:end] += w
            else:
                valid = T - t
                full_mask[:, t:T] += chunk_mask[:, :valid] * w[:valid][np.newaxis, :]
                weight[t:T] += w[:valid]

            t += hop_steps

        weight_safe = np.where(weight > 0, weight, 1.0)
        full_mask = full_mask / weight_safe[np.newaxis, :]
        return np.clip(full_mask, 0.0, 1.0)
    def _wavenet_denoise_waveform(self, model, audio, sr,
                                  segment_seconds=1.0,
                                  hop_seconds=0.5):
        audio = audio.astype(np.float32)
        n_samples = len(audio)

        segment_len = int(sr * segment_seconds)
        hop = int(sr * hop_seconds)

        if segment_len <= 0:
            segment_len = n_samples

        
        if n_samples <= segment_len:
            inp = audio[np.newaxis, :, np.newaxis]
            pred = model.predict(inp, verbose=0)[0, :, 0]
            if len(pred) > n_samples:
                pred = pred[:n_samples]
            elif len(pred) < n_samples:
                pred = np.pad(pred, (0, n_samples - len(pred)), mode="constant")
            return pred
        denoised = np.zeros(n_samples, dtype=np.float32)
        weight = np.zeros(n_samples, dtype=np.float32)

     
        win = np.hanning(segment_len).astype(np.float32)
        if segment_len == 1:
            win[:] = 1.0
        t = 0
        while t < n_samples:
            end = t + segment_len
            if end <= n_samples:
                chunk = audio[t:end]
                valid = segment_len
            else:
                pad_len = end - n_samples
                chunk = np.pad(audio[t:n_samples], (0, pad_len), mode="constant")
                valid = n_samples - t
            inp = chunk[np.newaxis, :, np.newaxis]  # (1, time, 1)
            pred = model.predict(inp, verbose=0)[0, :, 0]  # (time,)

            if valid < segment_len:
                w = np.ones(valid, dtype=np.float32)
            else:
                w = win
            if end <= n_samples:
                denoised[t:end] += pred[:valid] * w
                weight[t:end] += w
            else:
                denoised[t:n_samples] += pred[:valid] * w[:valid]
                weight[t:n_samples] += w[:valid]
            t += hop
        weight_safe = np.where(weight > 0, weight, 1.0)
        denoised = denoised / weight_safe
        return denoised



model_manager = AudioModelManager()



@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Audio Background Noise Remover',
        'models_loaded': list(model_manager.models.keys()),
        'tensorflow_version': tf.__version__,
        'timestamp': datetime.utcnow().isoformat()
    })

@app.route('/api/denoise-audio', methods=['POST'])
def denoise_audio():
    """Main audio denoising endpoint"""
    temp_input_file = None
    
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        file = request.files['audio']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        user_id = request.form.get('user_id', 'anonymous')
        strength = float(request.form.get('strength', 0.8))
        model_type = request.form.get('model_type', 'spectral_unet')
 
        if strength < 0.1 or strength > 1.0:
            return jsonify({'error': 'Strength must be between 0.1 and 1.0'}), 400
        
        if model_type not in model_manager.models:
            model_type = 'spectral_unet'  
        
       
        start_time = datetime.utcnow()
        
        unique_id = str(uuid.uuid4())
        temp_input_file = os.path.join(tempfile.gettempdir(), f'input_{unique_id}.wav')
        
      
        file.save(temp_input_file)
        
       
        denoised_audio, sr, noise_reduction_db = model_manager.denoise_audio(
            temp_input_file, model_type, strength
        )
        
        if denoised_audio is None:
            return jsonify({'error': 'Failed to process audio'}), 400
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        audio_duration = len(denoised_audio) / sr

        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, denoised_audio, sr, format='WAV')
        audio_buffer.seek(0)
  
        try:
            log_entry = ProcessingLog(
                user_id=user_id,
                original_filename=secure_filename(file.filename),
                model_type=model_type,
                noise_reduction_strength=strength,
                processing_time=processing_time,
                audio_duration=audio_duration,
                sample_rate=sr,
                noise_reduction_db=noise_reduction_db
            )
            db.session.add(log_entry)
            
          
            user = User.query.filter_by(user_id=user_id).first()
            if not user:
                user = User(
                    user_id=user_id,
                    total_processed=0,
                    total_duration_minutes=0.0,
                    avg_noise_reduction_db=0.0
                )
                db.session.add(user)
                db.session.flush() 
          
            if user.total_processed is None:
                user.total_processed = 0
            if user.total_duration_minutes is None:
                user.total_duration_minutes = 0.0
            if user.avg_noise_reduction_db is None:
                user.avg_noise_reduction_db = 0.0
            
            user.total_processed += 1
            user.total_duration_minutes += (audio_duration / 60)
            
            if user.total_processed > 0:
                user.avg_noise_reduction_db = (
                    (user.avg_noise_reduction_db * (user.total_processed - 1) + noise_reduction_db) 
                    / user.total_processed
                )
            
            db.session.commit()
            logger.info(f"Database updated successfully for user {user_id}")
            
        except Exception as db_error:
            logger.warning(f"Database logging failed: {db_error}")
            db.session.rollback()
        
        logger.info(f"Successfully processed audio for user {user_id} using {model_type}")
        
        return send_file(
            audio_buffer,
            mimetype='audio/wav',
            as_attachment=False,
            download_name=f'denoised_{secure_filename(file.filename)}'
        )
        
    except Exception as e:
        logger.error(f"Error in denoise-audio endpoint: {e}")
        return jsonify({'error': f'Processing failed. Please try again.'}), 500
        
    finally:

        if temp_input_file and os.path.exists(temp_input_file):
            try:
                os.unlink(temp_input_file)
            except Exception as e:
                logger.warning(f"Could not delete temp input file: {e}")

@app.route('/api/user-stats/<user_id>', methods=['GET'])
def get_user_stats(user_id):
    """Get user statistics"""
    try:
        user = User.query.filter_by(user_id=user_id).first()
        
        if not user:
            return jsonify({
                'processed_count': 0,
                'total_duration_minutes': 0.0,
                'avg_noise_reduction_db': 0.0,
                'member_since': None
            })
        
        processed_count = user.total_processed if user.total_processed is not None else 0
        duration_minutes = user.total_duration_minutes if user.total_duration_minutes is not None else 0.0
        avg_noise_reduction = user.avg_noise_reduction_db if user.avg_noise_reduction_db is not None else 0.0
        
        return jsonify({
            'processed_count': processed_count,
            'total_duration_minutes': round(duration_minutes, 2),
            'avg_noise_reduction_db': round(avg_noise_reduction, 2),
            'member_since': user.created_at.isoformat() if user.created_at else None
        })
        
    except Exception as e:
        logger.error(f"Error getting user stats: {e}")
        return jsonify({'error': 'Failed to retrieve stats'}), 500

@app.route('/api/processing-history/<user_id>', methods=['GET'])
def get_processing_history(user_id):
    """Get user's processing history"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        
        logs = ProcessingLog.query.filter_by(user_id=user_id)\
                                 .order_by(ProcessingLog.timestamp.desc())\
                                 .paginate(page=page, per_page=per_page, error_out=False)
        
        history = []
        for log in logs.items:
            processing_time = log.processing_time if log.processing_time is not None else 0.0
            duration = log.audio_duration if log.audio_duration is not None else 0.0
            sample_rate = log.sample_rate if log.sample_rate is not None else 0
            noise_reduction = log.noise_reduction_db if log.noise_reduction_db is not None else 0.0
            
            history.append({
                'id': log.id,
                'filename': log.original_filename,
                'model_type': log.model_type,
                'strength': log.noise_reduction_strength,
                'processing_time': round(processing_time, 2),
                'duration': round(duration, 2),
                'sample_rate': sample_rate,
                'noise_reduction_db': round(noise_reduction, 2),
                'timestamp': log.timestamp.isoformat() if log.timestamp else None
            })
        
        return jsonify({
            'history': history,
            'total': logs.total,
            'pages': logs.pages,
            'current_page': page,
            'has_next': logs.has_next,
            'has_prev': logs.has_prev
        })
        
    except Exception as e:
        logger.error(f"Error getting processing history: {e}")
        return jsonify({'error': 'Failed to retrieve history'}), 500

@app.route('/api/models', methods=['GET'])
def get_available_models():
    """Get list of available models"""
    return jsonify({
        'models': [
            {
                'name': 'spectral_unet',
                'display_name': 'Spectral U-Net',
                'description': 'Frequency domain noise removal - best for constant background noise',
                'speed': 'fast',
                'quality': 'good',
                'best_for': 'Constant background noise (AC, fan, hum)'
            },
            {
                'name': 'rnn_denoiser',
                'display_name': 'RNN Denoiser',
                'description': 'Neural network for time-varying noise patterns',
                'speed': 'medium',
                'quality': 'better',
                'best_for': 'Time-varying noise (traffic, crowds)'
            },
            {
                'name': 'wavenet',
                'display_name': 'WaveNet',
                'description': 'Advanced time-domain processing',
                'speed': 'slow',
                'quality': 'best',
                'best_for': 'Complex noise patterns and speech enhancement'
            }
        ]
    })

@app.route('/api/audio-info', methods=['POST'])
def get_audio_info():
    temp_file_path = None
    
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        file = request.files['audio']
        
    
        unique_id = str(uuid.uuid4())
        temp_file_path = os.path.join(tempfile.gettempdir(), f'audio_info_{unique_id}.wav')
        
        file.save(temp_file_path)
        audio, sr = model_manager.audio_processor.load_audio(temp_file_path)
        
        if audio is None:
            return jsonify({'error': 'Failed to load audio file'}), 400
        
        duration = len(audio) / sr
        
        return jsonify({
            'filename': secure_filename(file.filename),
            'duration': round(duration, 2),
            'sample_rate': sr,
            'channels': 1,  # We convert to mono
            'format': 'Processed as WAV mono'
        })
        
    except Exception as e:
        logger.error(f"Error getting audio info: {e}")
        return jsonify({'error': 'Failed to analyze audio'}), 500
        
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"Could not delete temp file: {e}")


def create_tables():
    """Create database tables"""
    with app.app_context():
        try:
            db.create_all()
            logger.info("Database tables created successfully")
            
            try:
                
                users_with_none = User.query.filter(
                    db.or_(
                        User.total_processed.is_(None),
                        User.total_duration_minutes.is_(None),
                        User.avg_noise_reduction_db.is_(None)
                    )
                ).all()
                
                for user in users_with_none:
                    if user.total_processed is None:
                        user.total_processed = 0
                    if user.total_duration_minutes is None:
                        user.total_duration_minutes = 0.0
                    if user.avg_noise_reduction_db is None:
                        user.avg_noise_reduction_db = 0.0
                
                if users_with_none:
                    db.session.commit()
                    logger.info(f"Fixed {len(users_with_none)} users with None values")
                    
            except Exception as migration_error:
                logger.warning(f"Migration failed: {migration_error}")
                db.session.rollback()
                
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")


if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    create_tables()

    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
