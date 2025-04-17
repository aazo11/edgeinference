#!/usr/bin/env python3
"""
WebLLM Benchmark browser utilities.
This module provides utility functions for running WebLLM benchmarks in a browser.
"""

import os
import sys
import time
import json
import socket
from urllib.request import urlopen
from urllib.error import URLError
import urllib.parse
from selenium.webdriver.chrome.options import Options

# Add the parent directory to the path so we can import from the benchmark_runner
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir.endswith('webllm'):
    # If running from inside the webllm directory
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)

# Import the frontend builder
from webllm.build_frontend import build_frontend

def wait_for_server(url, timeout=5):
    """Wait for a server to be available at the given URL."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with urlopen(url, timeout=1) as response:
                return True
        except URLError:
            time.sleep(0.1)
    return False

def find_free_port():
    """Find a free port on the system."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def run_browser_benchmark(test, debug=False, force_visible=False, batch_info=None):
    """
    Run the benchmark in a browser using Selenium.
    
    Args:
        test: A BenchmarkTest object containing the test parameters
        debug: Whether to run in debug mode (headless=False)
        force_visible: If True, will always show the browser window regardless of debug setting
        batch_info: Optional dictionary with batch information (batch_index, batch_total, batch_id)
        
    Returns:
        Tuple of (response, latency_ms, tokens_per_second)
    """
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from webdriver_manager.chrome import ChromeDriverManager
        import platform
    except ImportError:
        print("Selenium not installed. Please install with: pip install selenium webdriver-manager")
        return "Error: Selenium not installed", 0, 0
    
    # Build the frontend before running the benchmark
    print("Building frontend...")
    try:
        build_frontend()
        print("Frontend built successfully")
    except Exception as e:
        print(f"Error building frontend: {e}")
        import traceback
        traceback.print_exc()
        return f"Error building frontend: {e}", 0, 0
    
    # Set up Chrome options
    chrome_options = Options()
    
    # Basic options for stability
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu-driver-bug-workarounds")
    
    # Memory settings for large models
    chrome_options.add_argument("--js-flags=--expose-gc")
    chrome_options.add_argument("--js-flags=--max-old-space-size=8192")
    
    # UI settings
    chrome_options.add_argument("--disable-infobars")
    chrome_options.add_argument("--disable-extensions")
    
    # Enable WebGPU (critical for WebLLM)
    chrome_options.add_argument("--enable-features=Vulkan,UseSkiaRenderer")
    chrome_options.add_argument("--enable-unsafe-webgpu")
    chrome_options.add_argument("--enable-dawn-features=allow_unsafe_apis")
    chrome_options.add_argument("--enable-gpu-rasterization")
    chrome_options.add_argument("--enable-zero-copy")
    chrome_options.add_argument("--ignore-gpu-blocklist")
    
    # Force visibility for WebGPU compatibility
    if not debug and not force_visible:
        # We're not using headless mode because it often has issues with WebGPU
        # chrome_options.add_argument("--headless=new")
        pass
    
    # Get model name from environment variable or use default
    model_name = os.environ.get('WEBLLM_MODEL', 'DeepSeek-R1-Distill-Qwen-7B-q4f16_1-MLC')
    
    # Convert test object to a dictionary for JSON serialization
    test_params = {
        "test_id": test.id,
        "prompt": test.prompt,
        "max_tokens": test.max_tokens,
        "temperature": test.temperature,
        "expected_class": test.expected_class,
        "model_name": model_name
    }
    
    # Add batch information if provided
    if batch_info:
        test_params.update({
            "batch_index": batch_info.get("batch_index", 0),
            "batch_total": batch_info.get("batch_total", 1),
            "batch_id": batch_info.get("batch_id", "single-test")
        })
    
    # Encode the test parameters as a URL parameter
    encoded_params = urllib.parse.quote(json.dumps(test_params))
    
    # Get the path to the index.html file - check multiple possible locations
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check multiple possible locations for the HTML file, prioritizing the dist directory
    possible_paths = [
        os.path.join(current_dir, 'frontend', 'dist', 'index.html'),  # Primary location
        os.path.join(current_dir, 'dist', 'index.html'),
        os.path.join(current_dir, 'index.html'),
        os.path.join(current_dir, 'frontend', 'build', 'index.html')
    ]
    
    html_path = None
    for path in possible_paths:
        if os.path.exists(path):
            html_path = path
            print(f"Found HTML file at: {html_path}")
            break
    
    if not html_path:
        print("Error: Could not find index.html in any of the expected locations")
        print(f"Checked: {possible_paths}")
        
        # Try to build the frontend again if we couldn't find the HTML file
        try:
            print("Attempting to build frontend again...")
            build_frontend(force=True)
            
            # Check paths again after building
            for path in possible_paths:
                if os.path.exists(path):
                    html_path = path
                    print(f"Found HTML file at: {html_path} after rebuilding")
                    break
        except Exception as e:
            print(f"Error rebuilding frontend: {e}")
        
        if not html_path:
            return "Error: Could not find index.html", 0, 0
    
    url = f"file://{html_path}?testParams={encoded_params}"
    print(f"Starting browser with URL: {url}")
    
    # Initialize the WebDriver with proper ChromeDriver for the platform
    try:
        # Use the manual approach directly since the webdriver_manager is having issues
        import subprocess
        import shutil
        
        # Create a directory for ChromeDriver if it doesn't exist
        chromedriver_dir = os.path.join(os.path.expanduser("~"), ".chromedriver")
        if os.path.exists(chromedriver_dir):
            # Clean up any existing chromedriver directory to avoid issues
            try:
                # Remove existing chromedriver executable but keep the directory
                existing_driver = os.path.join(chromedriver_dir, "chromedriver")
                if os.path.exists(existing_driver):
                    os.remove(existing_driver)
                
                # Clean up any existing zip files
                zip_path = os.path.join(chromedriver_dir, "chromedriver.zip")
                if os.path.exists(zip_path):
                    os.remove(zip_path)
                
                # Clean up any extracted directories
                for item in os.listdir(chromedriver_dir):
                    item_path = os.path.join(chromedriver_dir, item)
                    if os.path.isdir(item_path) and "chromedriver" in item:
                        shutil.rmtree(item_path, ignore_errors=True)
                
                print("Cleaned up existing chromedriver files")
            except Exception as clean_error:
                print(f"Warning: Could not clean up chromedriver directory: {clean_error}")
        else:
            os.makedirs(chromedriver_dir, exist_ok=True)
        
        # Check if ChromeDriver is already downloaded
        chromedriver_path = os.path.join(chromedriver_dir, "chromedriver")
        
        # Check if we're on macOS ARM64 (Apple Silicon)
        is_mac_arm = platform.system() == 'Darwin' and platform.machine() == 'arm64'
        
        # Get Chrome version to match chromedriver version
        chrome_version = None
        try:
            # Try to get Chrome version from the command line
            if platform.system() == 'Darwin':  # macOS
                result = subprocess.run(
                    ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "--version"],
                    capture_output=True, text=True, check=False
                )
                if result.stdout:
                    chrome_version = result.stdout.split("Google Chrome ")[1].split(" ")[0]
                    print(f"Detected Chrome version: {chrome_version}")
            
            # If can't detect Chrome version, use a stable default
            if not chrome_version:
                chrome_version = "114.0.5735.90"  # Use a stable version as fallback
                print(f"Using default Chrome version: {chrome_version}")
                
            # Extract major version
            major_version = chrome_version.split('.')[0]
            
        except Exception as e:
            print(f"Error detecting Chrome version: {e}")
            # Use a stable version as fallback
            major_version = "114"
            print(f"Using default Chrome major version: {major_version}")
        
        # Force redownload if needed
        force_redownload = os.environ.get('FORCE_CHROMEDRIVER_DOWNLOAD', '').lower() in ('true', '1', 'yes')
        
        if not os.path.exists(chromedriver_path) or not os.access(chromedriver_path, os.X_OK) or force_redownload:
            print("Downloading ChromeDriver manually...")
            
            # For macOS ARM64, download the appropriate version
            if is_mac_arm:
                # URL for the ChromeDriver for macOS ARM64
                download_url = f"https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/{chrome_version}/mac-arm64/chromedriver-mac-arm64.zip"
            else:
                # URL for the ChromeDriver for macOS x64
                download_url = f"https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/{chrome_version}/mac-x64/chromedriver-mac-x64.zip"
            
            # Download and extract ChromeDriver
            zip_path = os.path.join(chromedriver_dir, "chromedriver.zip")
            
            try:
                print(f"Downloading ChromeDriver from: {download_url}")
                subprocess.run(["curl", "-L", download_url, "-o", zip_path], check=True)
                
                # Create a temporary extraction directory
                extract_dir = os.path.join(chromedriver_dir, "extracted")
                if os.path.exists(extract_dir):
                    shutil.rmtree(extract_dir, ignore_errors=True)
                os.makedirs(extract_dir, exist_ok=True)
                
                # Extract the zip file
                import zipfile
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                
                # Find the chromedriver executable in the extracted files
                chromedriver_found = False
                for root, dirs, files in os.walk(extract_dir):
                    for file in files:
                        if file == "chromedriver" or file == "chromedriver.exe":
                            extracted_path = os.path.join(root, file)
                            # Copy to the expected location 
                            shutil.copy2(extracted_path, chromedriver_path)
                            chromedriver_found = True
                            print(f"Found chromedriver at {extracted_path}, copied to {chromedriver_path}")
                            break
                    if chromedriver_found:
                        break
                
                if not chromedriver_found:
                    raise Exception("Could not find chromedriver executable in the extracted files")
                
                # Make it executable
                os.chmod(chromedriver_path, 0o755)
                print(f"ChromeDriver downloaded and extracted to {chromedriver_path}")
                
                # Clean up the temporary directory and zip file
                shutil.rmtree(extract_dir, ignore_errors=True)
                
            except Exception as download_error:
                print(f"Error downloading ChromeDriver for version {chrome_version}: {download_error}")
                print("Falling back to stable version...")
                
                # Fallback to a stable version that's known to work
                fallback_version = "114.0.5735.90"
                
                # Use direct links for stable version
                if is_mac_arm:
                    download_url = f"https://chromedriver.storage.googleapis.com/{fallback_version}/chromedriver_mac_arm64.zip"
                else:
                    download_url = f"https://chromedriver.storage.googleapis.com/{fallback_version}/chromedriver_mac64.zip"
                
                try:
                    print(f"Downloading fallback ChromeDriver from: {download_url}")
                    # Clean any failed previous attempt
                    if os.path.exists(zip_path):
                        os.remove(zip_path)
                        
                    # Download the fallback driver
                    subprocess.run(["curl", "-L", download_url, "-o", zip_path], check=True)
                    
                    # Create a fresh temporary extraction directory
                    extract_dir = os.path.join(chromedriver_dir, "extracted_fallback")
                    if os.path.exists(extract_dir):
                        shutil.rmtree(extract_dir, ignore_errors=True)
                    os.makedirs(extract_dir, exist_ok=True)
                    
                    # Extract using Python's zipfile
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_dir)
                    
                    # Find chromedriver in the extracted directory
                    chromedriver_found = False
                    for root, dirs, files in os.walk(extract_dir):
                        for file in files:
                            if file == "chromedriver" or file == "chromedriver.exe":
                                extracted_path = os.path.join(root, file)
                                shutil.copy2(extracted_path, chromedriver_path)
                                chromedriver_found = True
                                print(f"Found fallback chromedriver at {extracted_path}")
                                break
                        if chromedriver_found:
                            break
                    
                    if not chromedriver_found:
                        raise Exception("Could not find chromedriver in fallback archive")
                    
                    # Make it executable
                    os.chmod(chromedriver_path, 0o755)
                    print(f"Fallback ChromeDriver downloaded and extracted to {chromedriver_path}")
                    
                    # Clean up temporary files
                    shutil.rmtree(extract_dir, ignore_errors=True)
                    
                except Exception as fallback_error:
                    print(f"Error downloading fallback ChromeDriver: {fallback_error}")
                    raise fallback_error
        
        # Use the manually downloaded ChromeDriver
        service = Service(executable_path=chromedriver_path)
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
    except Exception as e:
        print(f"Error initializing ChromeDriver: {e}")
        import traceback
        traceback.print_exc()
        return f"Error initializing ChromeDriver: {e}", 0, 0
    
    try:
        # Navigate to the benchmark page
        driver.get(url)
        
        # Wait for the page to load
        print("Waiting for page to load...")
        
        # Add a more robust page load check with better error recovery
        page_loaded = False
        max_retries = 3
        retry_count = 0
        
        while not page_loaded and retry_count < max_retries:
            try:
                # Print some basic page details for debugging
                print(f"Current URL: {driver.current_url}")
                print(f"Title: {driver.title}")
                
                # Use a shorter timeout first and check for basic page load
                WebDriverWait(driver, 5).until(
                    lambda d: d.execute_script("return document.readyState") == "complete"
                )
                
                # Now check for elements that are actually present in the page
                # We know from the HTML that these elements exist
                WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.ID, "model-name-display"))
                )
                
                WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.ID, "status"))
                )
                
                WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.ID, "results"))
                )
                
                page_loaded = True
                print("Page loaded successfully!")
                
            except Exception as e:
                retry_count += 1
                print(f"Error waiting for page to load (try {retry_count}/{max_retries}): {e}")
                
                # Print more diagnostic information
                print("Trying to debug page load issue...")
                try:
                    # Try to get page source regardless of load state
                    page_source = driver.page_source
                    print(f"Current page source (first 500 chars): {page_source[:500]}...")
                    
                    # Check if there are any errors in the console log
                    try:
                        console_logs = driver.get_log('browser')
                        if console_logs:
                            print("Browser console logs:")
                            for log in console_logs[:10]:  # Show first 10 logs
                                print(f"  {log}")
                    except Exception as log_error:
                        print(f"Could not retrieve browser logs: {log_error}")
                    
                    # Give the page a bit more time to load
                    time.sleep(5)
                    
                except Exception as debug_error:
                    print(f"Error getting debug information: {debug_error}")
                
                if retry_count >= max_retries:
                    return f"Error: Page did not load correctly after {max_retries} tries: {e}", 0, 0
        
        # Check if auto-benchmark is working, if not, click the button
        print("Checking if auto-benchmark is running...")
        time.sleep(3)  # Give a moment for auto-benchmark to start
        
        status_element = driver.find_element(By.ID, "status")
        model_name_element = driver.find_element(By.ID, "model-name-display")
        
        # Check if the model name is displayed
        print(f"Model name displayed: {model_name_element.text}")
        
        # Check the status class to determine if it's loading
        if "loading" in status_element.get_attribute("class"):
            print("Auto-benchmark is running...")
        else:
            print("Status element found but not in loading state: " + status_element.get_attribute("class"))
            print("Status text: " + status_element.text)
            
            # Since there's no button to click, we need to wait for auto-benchmark to initialize
            # or trigger it with JavaScript
            try:
                # Try to trigger initialization via JavaScript
                driver.execute_script("""
                    if (window.initAutoBenchmark) {
                        console.log('Manually triggering initAutoBenchmark');
                        window.initAutoBenchmark();
                    } else {
                        console.log('initAutoBenchmark not found');
                    }
                """)
                print("Attempted to trigger benchmark via JavaScript")
            except Exception as js_error:
                print(f"Error executing JavaScript: {js_error}")
            
            # Wait again to see if status changes to loading
            time.sleep(5)
            status_element = driver.find_element(By.ID, "status")
            if "loading" in status_element.get_attribute("class"):
                print("Auto-benchmark is now running")
            else:
                print("Warning: Auto-benchmark may not be running. Will wait for completion anyway.")
        
        # Wait for the benchmark to complete (status changes to success or error)
        print("Waiting for benchmark to complete...")
        WebDriverWait(driver, 600).until(
            lambda d: "success" in d.find_element(By.ID, "status").get_attribute("class") or 
                     "error" in d.find_element(By.ID, "status").get_attribute("class")
        )
        
        # Wait for the completion marker to appear in the DOM
        print("Waiting for completion marker...")
        try:
            WebDriverWait(driver, 30).until(
                lambda d: d.find_element(By.ID, "benchmark-complete") is not None or
                          d.find_element(By.ID, "benchmark-error") is not None
            )
        except:
            print("Warning: Completion marker not found, but continuing...")
        
        # Get the results
        status_element = driver.find_element(By.ID, "status")
        results_element = driver.find_element(By.ID, "results")
        
        # Check if there was an error
        if "error" in status_element.get_attribute("class"):
            print(f"Benchmark error: {status_element.text}")
            return status_element.text, 0, 0
        
        # Try to get the result from the completion marker
        try:
            completion_element = driver.find_element(By.ID, "benchmark-complete")
            result_json = completion_element.get_attribute("data-result")
            if result_json:
                result = json.loads(result_json)
                print(f"Got result from completion marker: {result}")
                return result['response'], result['latency_ms'], result['tokens_per_second']
        except:
            print("Could not get result from completion marker, falling back to UI elements")
        
        # Extract the response text
        response = results_element.text
        
        # Extract metrics from the status text
        status_text = status_element.text
        print(f"Status: {status_text}")
        
        # Parse the metrics from the status text
        # Example: "Benchmark completed: Generated 100 tokens in 5000.00ms (20.00 tokens/sec)"
        try:
            latency_ms = float(status_text.split("in ")[1].split("ms")[0].strip())
            tokens_per_second = float(status_text.split("(")[1].split(" tokens/sec")[0].strip())
        except (IndexError, ValueError):
            print("Could not parse metrics from status text, using default values")
            latency_ms = 0
            tokens_per_second = 0
        
        return response, latency_ms, tokens_per_second
        
    except Exception as e:
        print(f"Error during browser benchmark: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}", 0, 0
        
    finally:
        # Close the browser
        driver.quit()

def run_browser_benchmark_batch(tests, debug=False, force_visible=False):
    """
    Run multiple benchmarks in a single browser session.
    
    Args:
        tests: A list of BenchmarkTest objects
        debug: Whether to run in debug mode (headless=False)
        force_visible: If True, will always show the browser window regardless of debug setting
        
    Returns:
        List of tuples, each containing (response, latency_ms, tokens_per_second) for a test
    """
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from webdriver_manager.chrome import ChromeDriverManager
        import platform
        import uuid
    except ImportError:
        print("Selenium not installed. Please install with: pip install selenium webdriver-manager")
        return [("Error: Selenium not installed", 0, 0)] * len(tests)
    
    # Build the frontend before running the benchmark
    print("Building frontend...")
    try:
        build_frontend()
        print("Frontend built successfully")
    except Exception as e:
        print(f"Error building frontend: {e}")
        import traceback
        traceback.print_exc()
        return [(f"Error building frontend: {e}", 0, 0)] * len(tests)
    
    # Set up Chrome options
    chrome_options = Options()
    
    # Basic options for stability
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu-driver-bug-workarounds")
    
    # Memory settings for large models
    chrome_options.add_argument("--js-flags=--expose-gc")
    chrome_options.add_argument("--js-flags=--max-old-space-size=8192")
    
    # UI settings
    chrome_options.add_argument("--disable-infobars")
    chrome_options.add_argument("--disable-extensions")
    
    # Enable WebGPU (critical for WebLLM)
    chrome_options.add_argument("--enable-features=Vulkan,UseSkiaRenderer")
    chrome_options.add_argument("--enable-unsafe-webgpu")
    chrome_options.add_argument("--enable-dawn-features=allow_unsafe_apis")
    chrome_options.add_argument("--enable-gpu-rasterization")
    chrome_options.add_argument("--enable-zero-copy")
    chrome_options.add_argument("--ignore-gpu-blocklist")
    
    # Force visibility for WebGPU compatibility
    if not debug and not force_visible:
        # We're not using headless mode because it often has issues with WebGPU
        # chrome_options.add_argument("--headless=new")
        pass
    
    # Get model name from environment variable or use default
    model_name = os.environ.get('WEBLLM_MODEL', 'DeepSeek-R1-Distill-Qwen-7B-q4f16_1-MLC')
    
    # Generate a unique batch ID for tracking
    batch_id = str(uuid.uuid4())
    print(f"Generated batch ID: {batch_id}")
    
    # Get the path to the index.html file - check multiple possible locations
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check multiple possible locations for the HTML file, prioritizing the dist directory
    possible_paths = [
        os.path.join(current_dir, 'frontend', 'dist', 'index.html'),  # Primary location
        os.path.join(current_dir, 'dist', 'index.html'),
        os.path.join(current_dir, 'index.html'),
        os.path.join(current_dir, 'frontend', 'build', 'index.html')
    ]
    
    html_path = None
    for path in possible_paths:
        if os.path.exists(path):
            html_path = path
            print(f"Found HTML file at: {html_path}")
            break
    
    if not html_path:
        print("Error: Could not find index.html in any of the expected locations")
        print(f"Checked: {possible_paths}")
        
        # Try to build the frontend again if we couldn't find the HTML file
        try:
            print("Attempting to build frontend again...")
            build_frontend(force=True)
            
            # Check paths again after building
            for path in possible_paths:
                if os.path.exists(path):
                    html_path = path
                    print(f"Found HTML file at: {html_path} after rebuilding")
                    break
        except Exception as e:
            print(f"Error rebuilding frontend: {e}")
        
        if not html_path:
            return [("Error: Could not find index.html", 0, 0)] * len(tests)
    
    # Initialize the WebDriver with proper ChromeDriver for the platform
    driver = None
    try:
        # Use the manual approach directly since the webdriver_manager is having issues
        import subprocess
        import shutil
        
        # Create a directory for ChromeDriver if it doesn't exist
        chromedriver_dir = os.path.join(os.path.expanduser("~"), ".chromedriver")
        if os.path.exists(chromedriver_dir):
            # Clean up any existing chromedriver directory to avoid issues
            try:
                # Remove existing chromedriver executable but keep the directory
                existing_driver = os.path.join(chromedriver_dir, "chromedriver")
                if os.path.exists(existing_driver):
                    os.remove(existing_driver)
                
                # Clean up any existing zip files
                zip_path = os.path.join(chromedriver_dir, "chromedriver.zip")
                if os.path.exists(zip_path):
                    os.remove(zip_path)
                
                # Clean up any extracted directories
                for item in os.listdir(chromedriver_dir):
                    item_path = os.path.join(chromedriver_dir, item)
                    if os.path.isdir(item_path) and "chromedriver" in item:
                        shutil.rmtree(item_path, ignore_errors=True)
                
                print("Cleaned up existing chromedriver files")
            except Exception as clean_error:
                print(f"Warning: Could not clean up chromedriver directory: {clean_error}")
        else:
            os.makedirs(chromedriver_dir, exist_ok=True)
        
        # Check if ChromeDriver is already downloaded
        chromedriver_path = os.path.join(chromedriver_dir, "chromedriver")
        
        # Check if we're on macOS ARM64 (Apple Silicon)
        is_mac_arm = platform.system() == 'Darwin' and platform.machine() == 'arm64'
        
        # Get Chrome version to match chromedriver version
        chrome_version = None
        try:
            # Try to get Chrome version from the command line
            if platform.system() == 'Darwin':  # macOS
                result = subprocess.run(
                    ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "--version"],
                    capture_output=True, text=True, check=False
                )
                if result.stdout:
                    chrome_version = result.stdout.split("Google Chrome ")[1].split(" ")[0]
                    print(f"Detected Chrome version: {chrome_version}")
            
            # If can't detect Chrome version, use a stable default
            if not chrome_version:
                chrome_version = "114.0.5735.90"  # Use a stable version as fallback
                print(f"Using default Chrome version: {chrome_version}")
                
            # Extract major version
            major_version = chrome_version.split('.')[0]
            
        except Exception as e:
            print(f"Error detecting Chrome version: {e}")
            # Use a stable version as fallback
            major_version = "114"
            print(f"Using default Chrome major version: {major_version}")
        
        # Force redownload if needed
        force_redownload = os.environ.get('FORCE_CHROMEDRIVER_DOWNLOAD', '').lower() in ('true', '1', 'yes')
        
        if not os.path.exists(chromedriver_path) or not os.access(chromedriver_path, os.X_OK) or force_redownload:
            # Download ChromeDriver (same implementation as in run_browser_benchmark)
            print("Downloading ChromeDriver manually...")
            # ... (same implementation for downloading ChromeDriver)
            # ... download code omitted for brevity (use the same code as in run_browser_benchmark)
            
            # For macOS ARM64, download the appropriate version
            if is_mac_arm:
                # URL for the ChromeDriver for macOS ARM64
                download_url = f"https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/{chrome_version}/mac-arm64/chromedriver-mac-arm64.zip"
            else:
                # URL for the ChromeDriver for macOS x64
                download_url = f"https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/{chrome_version}/mac-x64/chromedriver-mac-x64.zip"
            
            # Download and extract ChromeDriver
            zip_path = os.path.join(chromedriver_dir, "chromedriver.zip")
            
            try:
                print(f"Downloading ChromeDriver from: {download_url}")
                subprocess.run(["curl", "-L", download_url, "-o", zip_path], check=True)
                
                # Create a temporary extraction directory
                extract_dir = os.path.join(chromedriver_dir, "extracted")
                if os.path.exists(extract_dir):
                    shutil.rmtree(extract_dir, ignore_errors=True)
                os.makedirs(extract_dir, exist_ok=True)
                
                # Extract the zip file
                import zipfile
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                
                # Find the chromedriver executable in the extracted files
                chromedriver_found = False
                for root, dirs, files in os.walk(extract_dir):
                    for file in files:
                        if file == "chromedriver" or file == "chromedriver.exe":
                            extracted_path = os.path.join(root, file)
                            # Copy to the expected location 
                            shutil.copy2(extracted_path, chromedriver_path)
                            chromedriver_found = True
                            print(f"Found chromedriver at {extracted_path}, copied to {chromedriver_path}")
                            break
                    if chromedriver_found:
                        break
                
                if not chromedriver_found:
                    raise Exception("Could not find chromedriver executable in the extracted files")
                
                # Make it executable
                os.chmod(chromedriver_path, 0o755)
                print(f"ChromeDriver downloaded and extracted to {chromedriver_path}")
                
                # Clean up the temporary directory and zip file
                shutil.rmtree(extract_dir, ignore_errors=True)
                
            except Exception as download_error:
                print(f"Error downloading ChromeDriver for version {chrome_version}: {download_error}")
                print("Falling back to stable version...")
                
                # Fallback to a stable version that's known to work
                fallback_version = "114.0.5735.90"
                
                # Use direct links for stable version
                if is_mac_arm:
                    download_url = f"https://chromedriver.storage.googleapis.com/{fallback_version}/chromedriver_mac_arm64.zip"
                else:
                    download_url = f"https://chromedriver.storage.googleapis.com/{fallback_version}/chromedriver_mac64.zip"
                
                try:
                    print(f"Downloading fallback ChromeDriver from: {download_url}")
                    # Clean any failed previous attempt
                    if os.path.exists(zip_path):
                        os.remove(zip_path)
                        
                    # Download the fallback driver
                    subprocess.run(["curl", "-L", download_url, "-o", zip_path], check=True)
                    
                    # Create a fresh temporary extraction directory
                    extract_dir = os.path.join(chromedriver_dir, "extracted_fallback")
                    if os.path.exists(extract_dir):
                        shutil.rmtree(extract_dir, ignore_errors=True)
                    os.makedirs(extract_dir, exist_ok=True)
                    
                    # Extract using Python's zipfile
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_dir)
                    
                    # Find chromedriver in the extracted directory
                    chromedriver_found = False
                    for root, dirs, files in os.walk(extract_dir):
                        for file in files:
                            if file == "chromedriver" or file == "chromedriver.exe":
                                extracted_path = os.path.join(root, file)
                                shutil.copy2(extracted_path, chromedriver_path)
                                chromedriver_found = True
                                print(f"Found fallback chromedriver at {extracted_path}")
                                break
                        if chromedriver_found:
                            break
                    
                    if not chromedriver_found:
                        raise Exception("Could not find chromedriver in fallback archive")
                    
                    # Make it executable
                    os.chmod(chromedriver_path, 0o755)
                    print(f"Fallback ChromeDriver downloaded and extracted to {chromedriver_path}")
                    
                    # Clean up temporary files
                    shutil.rmtree(extract_dir, ignore_errors=True)
                    
                except Exception as fallback_error:
                    print(f"Error downloading fallback ChromeDriver: {fallback_error}")
                    raise fallback_error
            
        # Use the manually downloaded ChromeDriver
        service = Service(executable_path=chromedriver_path)
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
    except Exception as e:
        print(f"Error initializing ChromeDriver: {e}")
        import traceback
        traceback.print_exc()
        return [(f"Error initializing ChromeDriver: {e}", 0, 0)] * len(tests)
    
    results = []
    
    try:
        # For each test, create and run the benchmark in the same browser session
        for i, test in enumerate(tests):
            print(f"\nRunning test {i+1}/{len(tests)}: {test.id}")
            
            # Convert test object to a dictionary for JSON serialization
            test_params = {
                "test_id": test.id,
                "prompt": test.prompt,
                "max_tokens": test.max_tokens,
                "temperature": test.temperature,
                "expected_class": test.expected_class,
                "model_name": model_name,
                # Add batch information
                "batch_index": i,
                "batch_total": len(tests),
                "batch_id": batch_id
            }
            
            # Encode the test parameters as a URL parameter
            encoded_params = urllib.parse.quote(json.dumps(test_params))
            
            # Create the URL for this test
            url = f"file://{html_path}?testParams={encoded_params}"
            print(f"Navigating to URL: {url}")
            
            # Navigate to the benchmark page
            driver.get(url)
            
            # Wait for the page to load
            print("Waiting for page to load...")
            
            # Add a more robust page load check with better error recovery
            page_loaded = False
            max_retries = 3
            retry_count = 0
            
            while not page_loaded and retry_count < max_retries:
                try:
                    # Print some basic page details for debugging
                    print(f"Current URL: {driver.current_url}")
                    print(f"Title: {driver.title}")
                    
                    # Use a shorter timeout first and check for basic page load
                    WebDriverWait(driver, 5).until(
                        lambda d: d.execute_script("return document.readyState") == "complete"
                    )
                    
                    # Now check for elements that are actually present in the page
                    WebDriverWait(driver, 15).until(
                        EC.presence_of_element_located((By.ID, "model-name-display"))
                    )
                    
                    WebDriverWait(driver, 15).until(
                        EC.presence_of_element_located((By.ID, "status"))
                    )
                    
                    WebDriverWait(driver, 15).until(
                        EC.presence_of_element_located((By.ID, "results"))
                    )
                    
                    page_loaded = True
                    print("Page loaded successfully!")
                    
                except Exception as e:
                    retry_count += 1
                    print(f"Error waiting for page to load (try {retry_count}/{max_retries}): {e}")
                    
                    # Print more diagnostic information
                    print("Trying to debug page load issue...")
                    try:
                        # Try to get page source regardless of load state
                        page_source = driver.page_source
                        print(f"Current page source (first 500 chars): {page_source[:500]}...")
                        
                        # Check if there are any errors in the console log
                        try:
                            console_logs = driver.get_log('browser')
                            if console_logs:
                                print("Browser console logs:")
                                for log in console_logs[:10]:  # Show first 10 logs
                                    print(f"  {log}")
                        except Exception as log_error:
                            print(f"Could not retrieve browser logs: {log_error}")
                        
                        # Give the page a bit more time to load
                        time.sleep(5)
                        
                    except Exception as debug_error:
                        print(f"Error getting debug information: {debug_error}")
                    
                    if retry_count >= max_retries:
                        print(f"Error: Page did not load correctly after {max_retries} tries: {e}")
                        results.append((f"Error: Page did not load correctly: {e}", 0, 0))
                        continue  # Move to the next test
            
            if not page_loaded:
                continue  # Skip to the next test
            
            # Check if auto-benchmark is working, if not, click the button
            print("Checking if auto-benchmark is running...")
            time.sleep(3)  # Give a moment for auto-benchmark to start
            
            status_element = driver.find_element(By.ID, "status")
            model_name_element = driver.find_element(By.ID, "model-name-display")
            
            # Check if the model name is displayed
            print(f"Model name displayed: {model_name_element.text}")
            
            # Check the status class to determine if it's loading
            if "loading" in status_element.get_attribute("class"):
                print("Auto-benchmark is running...")
            else:
                print("Status element found but not in loading state: " + status_element.get_attribute("class"))
                print("Status text: " + status_element.text)
                
                # Try to trigger initialization via JavaScript
                try:
                    driver.execute_script("""
                        if (window.initAutoBenchmark) {
                            console.log('Manually triggering initAutoBenchmark');
                            window.initAutoBenchmark();
                        } else {
                            console.log('initAutoBenchmark not found');
                        }
                    """)
                    print("Attempted to trigger benchmark via JavaScript")
                except Exception as js_error:
                    print(f"Error executing JavaScript: {js_error}")
                
                # Wait again to see if status changes to loading
                time.sleep(5)
                status_element = driver.find_element(By.ID, "status")
                if "loading" in status_element.get_attribute("class"):
                    print("Auto-benchmark is now running")
                else:
                    print("Warning: Auto-benchmark may not be running. Will wait for completion anyway.")
            
            # Wait for the benchmark to complete (status changes to success or error)
            print("Waiting for benchmark to complete...")
            try:
                WebDriverWait(driver, 600).until(
                    lambda d: "success" in d.find_element(By.ID, "status").get_attribute("class") or 
                            "error" in d.find_element(By.ID, "status").get_attribute("class")
                )
            except Exception as timeout_error:
                print(f"Timeout waiting for benchmark to complete: {timeout_error}")
                results.append(("Error: Benchmark timed out", 0, 0))
                continue  # Move to the next test
            
            # Wait for the completion marker to appear in the DOM
            print("Waiting for completion marker...")
            try:
                WebDriverWait(driver, 30).until(
                    lambda d: d.find_element(By.ID, "benchmark-complete") is not None or
                            d.find_element(By.ID, "benchmark-error") is not None
                )
            except:
                print("Warning: Completion marker not found, but continuing...")
            
            # Get the results
            status_element = driver.find_element(By.ID, "status")
            results_element = driver.find_element(By.ID, "results")
            
            # Check if there was an error
            if "error" in status_element.get_attribute("class"):
                print(f"Benchmark error: {status_element.text}")
                results.append((status_element.text, 0, 0))
                continue  # Move to the next test
            
            # Try to get the result from the completion marker
            try:
                completion_element = driver.find_element(By.ID, "benchmark-complete")
                result_json = completion_element.get_attribute("data-result")
                if result_json:
                    result = json.loads(result_json)
                    print(f"Got result from completion marker: {result}")
                    results.append((result['response'], result['latency_ms'], result['tokens_per_second']))
                    print(f"Completed test {i+1}/{len(tests)}")
                    continue  # Move to the next test
            except:
                print("Could not get result from completion marker, falling back to UI elements")
            
            # Extract the response text
            response = results_element.text
            
            # Extract metrics from the status text
            status_text = status_element.text
            print(f"Status: {status_text}")
            
            # Parse the metrics from the status text
            # Example: "Benchmark completed: Generated 100 tokens in 5000.00ms (20.00 tokens/sec)"
            try:
                latency_ms = float(status_text.split("in ")[1].split("ms")[0].strip())
                tokens_per_second = float(status_text.split("(")[1].split(" tokens/sec")[0].strip())
            except (IndexError, ValueError):
                print("Could not parse metrics from status text, using default values")
                latency_ms = 0
                tokens_per_second = 0
            
            results.append((response, latency_ms, tokens_per_second))
            print(f"Completed test {i+1}/{len(tests)}")
        
        return results
        
    except Exception as e:
        print(f"Error during browser benchmark batch: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # For any remaining tests, add error results
        while len(results) < len(tests):
            results.append((f"Error: {str(e)}", 0, 0))
        
        return results
        
    finally:
        # Close the browser
        if driver:
            driver.quit()

# For testing without using the benchmark runner
if __name__ == "__main__":
    try:
        from benchmark.benchmark_runner import BenchmarkTest
    except ImportError:
        # Create a simple BenchmarkTest class for standalone testing
        print("Could not import BenchmarkTest from benchmark_runner, using local implementation")
        class BenchmarkTest:
            def __init__(self, id, prompt, max_tokens, temperature, expected_class, notes):
                self.id = id
                self.prompt = prompt
                self.max_tokens = max_tokens
                self.temperature = temperature
                self.expected_class = expected_class
                self.notes = notes
    
    # Run test
    print("Running WebLLM benchmark test using default browser...")
    
    # Create results directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    test = BenchmarkTest(
        id="test1",
        prompt="Explain the concept of quantum computing in simple terms.",
        max_tokens=512,
        temperature=0.0,
        expected_class="explanation",
        notes="Test"
    )
    
    response, latency, tokens_per_sec = run_browser_benchmark(test, debug=True)
    print(f"\nResponse: {response}")
    print(f"Latency: {latency}ms")
    print(f"Tokens per second: {tokens_per_sec}") 