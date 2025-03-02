import torch
from transformers import pipeline
import speech_recognition as sr
import os
import re
import time
import subprocess
from pydub import AudioSegment
from pydub.silence import split_on_silence


def convert_audio_to_wav(audio_file_path):
    """
    Convert audio file to WAV format using FFmpeg
    Parameters:
        audio_file_path (str): Path to the audio file
    Returns:
        str: Path to the converted WAV file
    """
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

    # If already WAV format, return the path
    if audio_file_path.lower().endswith('.wav'):
        return audio_file_path

    # Convert to WAV
    wav_path = audio_file_path.rsplit('.', 1)[0] + '.wav'
    print(f"Converting audio to WAV format: {wav_path}")

    try:
        # Use subprocess to call ffmpeg
        subprocess.run([
            'ffmpeg',
            '-i', audio_file_path,
            '-ar', '16000',  # Set sampling rate to 16kHz for better recognition
            '-ac', '1',  # Convert to mono
            '-y',  # Overwrite without asking
            wav_path
        ], check=True)
        print(f"Conversion successful: {wav_path}")
        return wav_path
    except subprocess.SubprocessError as e:
        print(f"Error converting audio: {e}")
        raise RuntimeError(f"FFmpeg conversion failed: {e}")


def transcribe_audio_file(wav_file_path, chunk_length_ms=60000):
    """
    Transcribe audio file by splitting into manageable chunks
    Parameters:
        wav_file_path (str): Path to the WAV file
        chunk_length_ms (int): Length of each chunk in milliseconds
    Returns:
        str: Transcribed text
    """
    print(f"Loading audio file: {wav_file_path}")

    try:
        # Load audio file using pydub
        audio = AudioSegment.from_wav(wav_file_path)

        # Initialize recognizer
        recognizer = sr.Recognizer()

        # Adjust recognition parameters
        recognizer.energy_threshold = 300
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 0.8

        # Split audio into chunks based on silence or fixed size
        print("Splitting audio into chunks for processing...")

        # Try to split on silence first
        chunks = split_on_silence(
            audio,
            min_silence_len=500,
            silence_thresh=audio.dBFS - 16,
            keep_silence=500
        )

        # If no silence-based chunks or too few, use fixed-size chunks
        if len(chunks) < 3:
            print("Few silence-based chunks found, using fixed-size chunks instead")
            chunk_length = chunk_length_ms  # 60 seconds per chunk
            chunks = [audio[i:i + chunk_length] for i in range(0, len(audio), chunk_length)]

        print(f"Processing {len(chunks)} audio chunks...")

        # Process each chunk
        whole_text = ""
        for i, chunk in enumerate(chunks):
            # Progress update
            print(f"Processing chunk {i + 1}/{len(chunks)}...")

            # Export chunk to temporary file
            chunk_filename = f"temp_chunk_{i}.wav"
            chunk.export(chunk_filename, format="wav")

            # Process with retry mechanism
            retry_count = 0
            max_retries = 3
            success = False

            while retry_count < max_retries and not success:
                try:
                    with sr.AudioFile(chunk_filename) as source:
                        audio_data = recognizer.record(source)
                        text = recognizer.recognize_google(audio_data)
                        whole_text += text + " "
                        success = True
                except sr.UnknownValueError:
                    print(f"Chunk {i + 1}: Speech not recognized")
                    # Just continue without text for this chunk
                    success = True
                except sr.RequestError as e:
                    retry_count += 1
                    print(f"Chunk {i + 1}: API request error ({retry_count}/{max_retries}): {e}")
                    time.sleep(2)  # Wait before retrying
                except Exception as e:
                    print(f"Chunk {i + 1}: Unexpected error: {e}")
                    retry_count += 1
                    time.sleep(2)

            # Clean up temporary file
            if os.path.exists(chunk_filename):
                os.remove(chunk_filename)

            if not success:
                print(f"Failed to process chunk {i + 1} after {max_retries} retries")

        return whole_text.strip()

    except Exception as e:
        print(f"Error in transcription process: {e}")
        return f"Error transcribing audio: {str(e)}"


def clean_text(text):
    """
    Clean and preprocess the transcribed text
    Parameters:
        text (str): Raw transcribed text
    Returns:
        str: Cleaned text
    """
    if not text or text.strip() == "":
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^a-z0-9\s.,?!]', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def summarize_with_pytorch(text):
    """
    Summarize text using PyTorch models
    Parameters:
        text (str): Text to summarize
    Returns:
        str: Summarized text
    """
    # Check if text is too short or empty
    if not text or len(text.split()) < 10:
        return "Text too short to summarize or no transcription available."

    try:
        print("Loading summarization model...")

        # Use PyTorch pipeline
        summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            framework="pt"
        )

        # Handle text length
        max_input_length = 1024  # BART model limit
        if len(text.split()) > max_input_length:
            print("Text too long, breaking into segments...")

            # Break into segments
            words = text.split()
            segment_size = max_input_length - 100  # Leave some buffer
            segments = [' '.join(words[i:i + segment_size]) for i in range(0, len(words), segment_size)]

            # Summarize each segment
            summaries = []
            for i, segment in enumerate(segments):
                print(f"Summarizing segment {i + 1}/{len(segments)}...")
                if len(segment.split()) < 50:  # Skip very short segments
                    continue

                # Calculate appropriate max_length based on input length
                input_length = len(segment.split())
                max_length = min(150, input_length // 2)
                min_length = min(30, max_length // 2)

                try:
                    summary = summarizer(segment, max_length=max_length, min_length=min_length, do_sample=False)
                    summaries.append(summary[0]['summary_text'])
                except Exception as e:
                    print(f"Error summarizing segment {i + 1}: {e}")

            # Combine segment summaries
            combined_summary = ' '.join(summaries)

            # If combined summary is still long, summarize it again
            if len(combined_summary.split()) > max_input_length:
                print("Creating final summary of summaries...")
                final_summary = summarizer(
                    combined_summary[:max_input_length],
                    max_length=300,
                    min_length=75,
                    do_sample=False
                )
                return final_summary[0]['summary_text']

            return combined_summary
        else:
            # For shorter texts, summarize directly
            input_length = len(text.split())
            max_length = min(150, input_length // 2)
            min_length = min(30, max_length // 2)

            summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
            return summary[0]['summary_text']

    except Exception as e:
        print(f"Error during summarization: {e}")

        # Try a different approach - extractive summarization
        try:
            print("Attempting extractive summarization as fallback...")
            from summarizer import Summarizer
            model = Summarizer()

            # Extractive summary
            return model(text, ratio=0.2)  # Extract ~20% of the original text
        except Exception as e2:
            print(f"Fallback summarization failed: {e2}")

            # Very simple fallback - just return the first few sentences
            sentences = re.split(r'[.!?]+', text)
            return ' '.join(sentences[:3]) + "..."


def podcast_to_summary(audio_file_path):
    """
    Main function to convert podcast to summary
    Parameters:
        audio_file_path (str): Path to the podcast audio file
    Returns:
        str: Summarized podcast content
    """
    try:
        # Step 1: Convert audio to WAV format
        wav_path = convert_audio_to_wav(audio_file_path)

        # Step 2: Transcribe the audio
        print("Transcribing audio...")
        transcribed_text = transcribe_audio_file(wav_path)

        # Save transcription for reference
        with open("podcast_transcription.txt", "w") as f:
            f.write(transcribed_text)
        print("Transcription saved to podcast_transcription.txt")

        if not transcribed_text or transcribed_text.strip() == "":
            return "Could not generate summary: No text was transcribed from the audio file."

        # Step 3: Clean the text
        print("Cleaning transcribed text...")
        cleaned_text = clean_text(transcribed_text)

        # Step 4: Generate summary
        print("Generating summary...")
        summary = summarize_with_pytorch(cleaned_text)

        return summary

    except Exception as e:
        print(f"Error in podcast processing pipeline: {e}")
        return f"Error processing podcast: {str(e)}"


if __name__ == "__main__":
    # Example usage
    audio_file = "../data/podcast1.mp3"  # Use your actual path

    # Make sure the file exists
    if not os.path.exists(audio_file):
        print(f"Error: File not found - {audio_file}")
        exit(1)

    # Get summary
    summary = podcast_to_summary(audio_file)

    print("\n=== PODCAST SUMMARY ===\n")
    print(summary)

    # Save summary to file
    with open("podcast_summary.txt", "w") as f:
        f.write(summary)

    print("\nSummary saved to podcast_summary.txt")