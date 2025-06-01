import pytest
import os
import sys
import subprocess
import platform
from scipy.io import wavfile
import numpy as np
from unittest.mock import patch # For more complex/direct patching if needed by test structure

# Ensure the examples directory is in the path to import create_sample_audio
# This assumes 'tests' and 'examples' are siblings in the project structure
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
examples_dir = os.path.join(project_root, "examples")
sys.path.insert(0, examples_dir)
# Also add project root for compare_audio.py if it's directly in the root
sys.path.insert(0, project_root)

# Direct import - if this fails, pytest collection will show the error.
from example_usage import create_sample_audio


# --- Constants for tests ---
TEST_SAMPLE_RATE = 44100
TEST_DURATION_SHORT = 0.1  # seconds, for quick tests
TEST_FREQUENCY = 440  # Hz

# Path to compare_audio.py script and import its main logic
COMPARE_AUDIO_SCRIPT_PATH = os.path.join(project_root, "compare_audio.py")
try:
    # This assumes compare_audio.py is in project_root, which was added to sys.path
    from compare_audio import main_logic as compare_audio_main_logic
    compare_audio_imported = True
except ImportError as e:
    print(f"DEBUG: Failed to import main_logic from compare_audio: {e}")
    compare_audio_main_logic = None
    compare_audio_imported = False


# --- Helper Functions ---
def is_wav_file(filepath):
    if not os.path.exists(filepath):
        return False
    try:
        rate, data = wavfile.read(filepath)
        return rate > 0 and len(data) > 0
    except Exception:
        return False

# --- Tests for create_sample_audio ---
# @pytest.mark.skipif(create_sample_audio is None, reason="create_sample_audio could not be imported") # No longer needed with direct import
@pytest.mark.parametrize("waveform", ['sine', 'square', 'sawtooth', 'noise'])
def test_create_sample_audio_waveforms(tmp_path, waveform):
    """Test generation of different waveforms."""
    output_filename = tmp_path / f"test_{waveform}.wav"

    created_file = create_sample_audio(
        str(output_filename),
        duration=TEST_DURATION_SHORT,
        sample_rate=TEST_SAMPLE_RATE,
        waveform=waveform,
        frequency=TEST_FREQUENCY
    )

    assert created_file == str(output_filename), "Output path should match input path"
    assert os.path.exists(created_file), f"{waveform} WAV file was not created"

    try:
        rate, data = wavfile.read(created_file)
    except Exception as e:
        pytest.fail(f"Failed to read {waveform} WAV file {created_file}: {e}")

    assert rate == TEST_SAMPLE_RATE, f"Sample rate for {waveform} incorrect"

    expected_samples = int(TEST_DURATION_SHORT * TEST_SAMPLE_RATE)
    # Allow for slight variations in sample length due to linspace or other generation artifacts
    assert abs(len(data) - expected_samples) <= 10, \
        f"Duration for {waveform} incorrect. Expected ~{expected_samples} samples, got {len(data)}"

# @pytest.mark.skipif(create_sample_audio is None, reason="create_sample_audio could not be imported") # No longer needed
def test_create_sample_audio_default_params(tmp_path):
    """Test create_sample_audio with default parameters."""
    output_filename = tmp_path / "default_audio.wav"
    # Using a very short duration to override default for speed
    created_file = create_sample_audio(str(output_filename), duration=TEST_DURATION_SHORT)
    assert os.path.exists(created_file)
    assert is_wav_file(created_file)

# --- Tests for compare_audio.py ---

# Helper to get expected player command (copied and adapted from compare_audio.py for test verification)
def get_expected_player_command_for_os(audio_file_path):
    system = platform.system()
    abs_path = os.path.abspath(audio_file_path)
    if system == "Linux":
        return ["aplay", abs_path]
    elif system == "Darwin": # macOS
        return ["afplay", abs_path]
    elif system == "Windows":
        return ["powershell", "-c", f"(New-Object Media.SoundPlayer '{abs_path}').PlaySync()"]
    return None

# @pytest.mark.skipif(create_sample_audio is None, reason="create_sample_audio could not be imported or compare_audio.py not found") # No longer needed for create_sample_audio part
def test_compare_audio_script_invocation_and_analysis(tmp_path, capsys):
    """Test if compare_audio.py can be invoked, analyzes files, and exits cleanly (menu not tested here)."""
    original_wav = tmp_path / "original.wav"
    watermarked_wav = tmp_path / "watermarked.wav"

    create_sample_audio(str(original_wav), duration=TEST_DURATION_SHORT, waveform='sine', frequency=220)
    create_sample_audio(str(watermarked_wav), duration=TEST_DURATION_SHORT, waveform='sine', frequency=440) # Different freq

    assert os.path.exists(COMPARE_AUDIO_SCRIPT_PATH), f"compare_audio.py not found at {COMPARE_AUDIO_SCRIPT_PATH}"

    # We expect compare_audio.py to print analysis and then wait for input.
    # For this test, we're only checking if it starts, analyzes, and would then wait for menu input.
    # We'll send "3" (Exit) to stdin to make it terminate cleanly after analysis.
    process = subprocess.run(
        [sys.executable, COMPARE_AUDIO_SCRIPT_PATH, str(original_wav), str(watermarked_wav)],
        input="3\n",  # Provide "3" then newline to exit the menu
        text=True,
        capture_output=True,
        timeout=10 # Add a timeout
    )

    assert process.returncode == 0, f"compare_audio.py exited with {process.returncode}. Stderr: {process.stderr}"

    # Check if analysis JSON is printed (basic check)
    assert "Analyzing original audio..." in process.stdout
    assert "Analyzing watermarked audio..." in process.stdout
    assert "\"carrier_energy\":" in process.stdout # A key from the analysis output
    assert "\"file_path\": " in process.stdout # Another key

@pytest.mark.skipif(not compare_audio_imported, reason="compare_audio.main_logic could not be imported")
def test_compare_audio_playback_mocked_player_found(tmp_path, mocker):
    """Test compare_audio.main_logic playback when player is FOUND."""
    original_wav = tmp_path / "original_mock.wav"
    watermarked_wav = tmp_path / "watermarked_mock.wav"

    create_sample_audio(str(original_wav), duration=TEST_DURATION_SHORT, waveform='sine', frequency=300)
    create_sample_audio(str(watermarked_wav), duration=TEST_DURATION_SHORT, waveform='sine', frequency=600)

    # Mock external dependencies of compare_audio module
    # shutil.which is used in get_player_command
    mocked_shutil_which = mocker.patch('compare_audio.shutil.which', return_value='/fake/player_path')
    # platform.system is used in get_player_command
    mocked_platform_system = mocker.patch('compare_audio.platform.system', return_value='Linux') # Example OS
    # subprocess.run is used in play_audio
    mocked_subprocess_run = mocker.patch('compare_audio.subprocess.run')
    # builtins.input is used for menu choices
    mocked_input = mocker.patch('builtins.input', side_effect=['1', '3']) # Play original, then Exit

    # Call the main_logic directly
    return_code = compare_audio_main_logic(str(original_wav), str(watermarked_wav))

    assert return_code == 0, "main_logic should return 0 on success"
    mocked_input.assert_any_call("Enter your choice (1-3): ")
    mocked_shutil_which.assert_called_once_with('aplay') # Because we mocked platform.system to 'Linux'
    mocked_platform_system.assert_called() # ensure it was checked

    expected_player_cmd = ['/fake/player_path', os.path.abspath(str(original_wav))]
    # Check if subprocess.run was called with the expected command for playback
    # We need to be careful with how os.abspath might be called inside play_audio
    # The get_player_command in compare_audio.py does *not* use abspath for the player path itself,
    # but it does for the audio file path on Windows. Let's adjust.
    if mocked_platform_system.return_value == "Windows":
         expected_player_cmd = ['/fake/player_path', '-c', f"(New-Object Media.SoundPlayer '{os.path.abspath(str(original_wav))}').PlaySync()"]
    else: # Linux or Darwin
        expected_player_cmd = ['/fake/player_path', str(original_wav)]


    mocked_subprocess_run.assert_called_once_with(expected_player_cmd, check=True, capture_output=True)

@pytest.mark.skipif(not compare_audio_imported, reason="compare_audio.main_logic could not be imported")
def test_compare_audio_playback_mocked_player_not_found(tmp_path, mocker, capsys):
    """Test compare_audio.main_logic playback when player is NOT FOUND."""
    original_wav = tmp_path / "original_notfound.wav"
    watermarked_wav = tmp_path / "watermarked_notfound.wav"

    create_sample_audio(str(original_wav), duration=TEST_DURATION_SHORT, waveform='sawtooth')
    create_sample_audio(str(watermarked_wav), duration=TEST_DURATION_SHORT, waveform='square')

    mocker.patch('compare_audio.shutil.which', return_value=None) # Simulate player not found
    mocker.patch('compare_audio.platform.system', return_value='Linux') # Example OS
    mocked_subprocess_run = mocker.patch('compare_audio.subprocess.run') # For playback
    mocked_input = mocker.patch('builtins.input', side_effect=['1', '3']) # Attempt to play, then Exit

    return_code = compare_audio_main_logic(str(original_wav), str(watermarked_wav))

    assert return_code == 0, "main_logic should still return 0 if player not found but exits cleanly"
    mocked_input.assert_any_call("Enter your choice (1-3): ")
    mocked_subprocess_run.assert_not_called() # Playback should not be attempted

    captured = capsys.readouterr()
    assert "Audio player 'aplay' not found." in captured.out # Check for the warning message

if __name__ == '__main__':
    # This allows running pytest via "python tests/test_audio_tools.py"
    pytest.main([__file__])
