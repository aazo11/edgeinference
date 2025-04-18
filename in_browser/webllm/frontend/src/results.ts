import { BenchmarkResult } from './types';
import { debug, getUrlParams } from './utils';

/**
 * Function to save results to a file
 */
export async function saveResultsToFile(results: BenchmarkResult): Promise<boolean> {
  const { resultFile, saveEndpoint } = getUrlParams();
  
  if (!resultFile) {
    debug('No result file specified, not saving results');
    return false;
  }
  
  try {
    debug(`Saving results to file: ${resultFile}`);
    
    // Create a JSON string of the results
    const jsonData = JSON.stringify(results);
    
    // If a save endpoint is provided, use it
    if (saveEndpoint) {
      debug(`Using save endpoint: ${saveEndpoint}`);
      
      try {
        // Use fetch with POST to send the results to the server
        const response = await fetch(saveEndpoint, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: jsonData
        });
        
        if (response.ok) {
          debug('Results saved successfully via endpoint');
          return true;
        } else {
          const errorText = await response.text();
          throw new Error(`Server responded with ${response.status}: ${errorText}`);
        }
      } catch (error) {
        debug(`Error saving results via endpoint: ${(error as Error).message}`);
        console.error('Error saving results via endpoint:', error);
        
        // Fall back to download method if endpoint fails
        debug('Falling back to download method');
      }
    }
    
    // Default download method (fallback)
    // Create a Blob with the JSON data
    const blob = new Blob([jsonData], { type: 'application/json' });
    
    // Create a download link and trigger it
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = resultFile.split('/').pop() || 'benchmark_results.json';
    a.style.display = 'none';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    
    debug('Results saved successfully via download');
    return true;
  } catch (error) {
    debug(`Error saving results: ${(error as Error).message}`);
    console.error('Error saving results:', error);
    return false;
  }
} 