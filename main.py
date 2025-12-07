import numpy as np
from scipy.signal import stft
import SoapySDR
from SoapySDR import *
import cv2
from torchvision import transforms
from PIL import Image
import torch
import time

class jamming_detector:
    def __init__(self, sample_rate=1.8e6, center_freq=1575.42e6, gain=40, samples=1000000):
        # Initialize SDR device
        devices = SoapySDR.Device.enumerate()
        self.sdr = SoapySDR.Device(devices[0])

        self.sample_rate = sample_rate
        self.center_freq = center_freq
        self.gain = gain
        self.samples = samples
        self.configure_sdr()
        self.load_model()

    def configure_sdr(self):
        # Set sample rate and frequency
        self.sdr.setSampleRate(SOAPY_SDR_RX, 0, self.sample_rate)
        self.sdr.setFrequency(SOAPY_SDR_RX, 0, self.center_freq)

        # Set gain
        if self.gain > 0:
            self.sdr.setGainMode(SOAPY_SDR_RX, 0, False)
            self.sdr.setGain(SOAPY_SDR_RX, 0, self.gain)
        else:
            self.sdr.setGainMode(SOAPY_SDR_RX, 0, True)
        SoapySDR.setLogLevel(SOAPY_SDR_WARNING)
    
    def get_spectrogram(self):
        rxStream = self.sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
        self.sdr.activateStream(rxStream)
        num_samples = self.samples
        buff = np.empty(num_samples, dtype=np.complex64)
        total_received = 0

        # Rx samples in loop
        while total_received < num_samples:
            sr = self.sdr.readStream(rxStream, [buff[total_received:]], num_samples - total_received)
            if sr.ret > 0:
                total_received += sr.ret
            else:
                print("readStream error:", sr)
                break

        self.sdr.deactivateStream(rxStream)
        self.sdr.closeStream(rxStream)

        # STFT
        f, t, Zxx = stft(buff, fs=self.sample_rate, nperseg=1024, return_onesided=False)
        return Zxx
    
    def spectrogram_to_img(self, Zxx):
        # Obliczenie amplitudy (moduł)
        amplitude = np.abs(Zxx)
        # Przeliczenie na dB, omijamy log(0)
        amplitude_db = 20 * np.log10(amplitude + 1e-10)
        # Zakres amplitudy
        amplitude_db = np.clip(amplitude_db, -120, 0)
        # Normalizacja do 8 bitowego zakresu [0, 255]
        amplitude_norm = ((amplitude_db + 120) / 120) * 255
        gray_image = amplitude_norm.astype(np.uint8)
        # Zamiana osi żeby wyglądało jak w testach
        gray_image = gray_image.T  # transpozycja macierzy
        # Przycięcie na interesujący nas fragment
        desired_width = 2048
        desired_height = 1024
        gray_image_resized = cv2.resize(gray_image, (desired_width, desired_height), interpolation=cv2.INTER_LINEAR)
        # Zamiana góry obrazu z dołem
        gray_image_flipped = cv2.flip(gray_image_resized, 0)
        pil_img = Image.fromarray(gray_image_flipped)
        transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # ResNet expects 3-channel input
        transforms.Resize((224, 224)),  # ResNet18 default
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet normalization
                            [0.229, 0.224, 0.225])
        ])
        image_tensor = transform(pil_img).unsqueeze(0)
        return image_tensor
    
    def load_model(self, model_path='resnet18_jamming_detector.pth', device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Method 1: Load the full model (easiest)
        model = checkpoint['model']
        model = model.to(device)
        model.eval()
        
        print(f"Model loaded successfully!")
        print(f"Final Validation Accuracy: {checkpoint['final_val_accuracy']:.2f}%")
        print(f"Best Validation Accuracy: {checkpoint['best_val_accuracy']:.2f}% (Epoch {checkpoint['best_epoch']})")
        
        self.model = model
        self.checkpoint = checkpoint

    def predict_image(self, device=None, threshold=0.2):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Preprocess image
        image_tensor = self.spectrogram_to_img(self.get_spectrogram())
        image_tensor = image_tensor.to(device)
        
        # Run inference
        with torch.no_grad():
            output = self.model(image_tensor)
            #confidence = torch.sigmoid(output[0]).item()
            confidence = output.squeeze()
            print(confidence)
            prediction = 'no_jamming' if confidence > threshold else 'jamming'
        
        return prediction, confidence

if __name__ == "__main__":
    detector = jamming_detector()
    detector.load_model()
    times = []
    for i in range(100):
        start = time.perf_counter()
        prediction, confidence = detector.predict_image()
        end = time.perf_counter()
        times.append(end - start)
        print(f"Prediction: {prediction}, Confidence: {confidence:.4f}")
        print(f"Run {i+1}: {times[-1]:.6f} sec")

    avg_time = sum(times)/len(times)
    print(f"\nAverage prediction time: {avg_time:.6f} sec")