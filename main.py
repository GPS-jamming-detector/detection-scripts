import numpy as np
from scipy.signal import stft
import SoapySDR
from SoapySDR import *
from torchvision import transforms
from PIL import Image
import torch
import time
import sys
import csv
import datetime
from datetime import timezone

class jamming_detector:
    def __init__(self, fname, sample_rate=1.8e6, center_freq=1575.42e6, gain=40, samples=1000000):
        # Initialize SDR device
        devices = SoapySDR.Device.enumerate()
        self.sdr = SoapySDR.Device(devices[0])
        self.fname = fname
        self.detection_counter = 0
        self.sample_rate = sample_rate
        self.center_freq = center_freq
        self.gain = gain
        self.samples = samples
        self.sample_buff = np.empty(self.samples, dtype=np.complex64)
        self.configure_sdr()
        self.load_model()

    def __del__(self):
        self.sdr.deactivateStream(self.rxStream)
        self.sdr.closeStream(self.rxStream)

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
        self.rxStream = self.sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
        self.sdr.activateStream(self.rxStream)
        SoapySDR.setLogLevel(SOAPY_SDR_WARNING)
    
    def get_spectrogram(self):
        total_received = 0
        # Rx samples in loop
        while total_received < self.samples:
            sr = self.sdr.readStream(self.rxStream, [self.sample_buff[total_received:]], self.samples - total_received)
            if sr.ret > 0:
                total_received += sr.ret
            else:
                print("readStream error:", sr)
                break

        # STFT
        f, t, Zxx = stft(self.sample_buff, fs=self.sample_rate, nperseg=1024, return_onesided=False)
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
        # Zamiana góry obrazu z dołem
        pil_img = Image.fromarray(gray_image)
        img_name = "{}_{}.png".format(self.fname, self.detection_counter)
        self.detection_counter += 1
        pil_img.save(img_name)
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
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Method 1: Load the full model (easiest)
        model = checkpoint['model']
        model = model.to(self.device)
        model.eval()
        
        print(f"Model loaded successfully!")
        print(f"Final Validation Accuracy: {checkpoint['final_val_accuracy']:.2f}%")
        print(f"Best Validation Accuracy: {checkpoint['best_val_accuracy']:.2f}% (Epoch {checkpoint['best_epoch']})")
        
        self.model = model
        self.checkpoint = checkpoint

    def predict_image(self, threshold=0.3):
        
        # Preprocess image
        spectrogram = self.get_spectrogram()
        image_tensor = self.spectrogram_to_img(spectrogram)
        image_tensor = image_tensor.to(self.device)
        # Run inference
        with torch.no_grad():
            output = self.model(image_tensor)
            #confidence = torch.sigmoid(output[0]).item()
            confidence = output.squeeze()
            prediction = 'no_jamming' if confidence > threshold else 'jamming'

        return prediction, confidence

if __name__ == "__main__":
    out_file = sys.argv[1] + ".csv"
    dt = datetime.datetime.now(timezone.utc)
    a = dt.replace(tzinfo=timezone.utc)
    # Initialize detector
    detector = jamming_detector(fname=sys.argv[1])
    detector.load_model()

    # Open CSV and write header if new
    write_header = False
    try:
        with open(out_file, "x"):
            write_header = True
    except FileExistsError:
        pass  # File exists, append rows only

    with open(out_file, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)

        if write_header:
            writer.writerow(["timestamp", "runtime_sec", "prediction", "confidence"])

        print(f"Logging predictions to {out_file}. Press Ctrl+C to stop.\n")
        input("Press Enter to continue...")

        try:
            while True:
                start = time.perf_counter()
                prediction, confidence = detector.predict_image()
                end = time.perf_counter()

                runtime = end - start
                ts = a.timestamp()

                writer.writerow([ts, f"{runtime:.6f}", prediction, f"{confidence:.4f}"])
                csvfile.flush()

                print(f"{ts} | Pred: {prediction}, Conf: {confidence:.4f}, Time: {runtime:.6f}s")

        except KeyboardInterrupt:
            print("\nStopped by user. Exiting gracefully.")