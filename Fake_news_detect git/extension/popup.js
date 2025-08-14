document.addEventListener('DOMContentLoaded', () => {
  const analyzeBtn = document.getElementById('analyzeBtn');
  const resultDiv = document.getElementById('result');

  analyzeBtn.addEventListener('click', async () => {
    resultDiv.textContent = 'Analyzing...';

    // Get the current active tab
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    // Inject the content script to get the page's text
    const injectionResults = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: getPageText,
    });

    const pageText = injectionResults[0].result;

    if (!pageText || pageText.length < 100) {
      resultDiv.textContent = 'Not enough text found on page.';
      return;
    }
    
    // Call the backend API
    try {
      const response = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: pageText }),
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data = await response.json();
      resultDiv.textContent = `Prediction: ${data.prediction}`;

    } catch (error) {
      console.error('Error:', error);
      resultDiv.textContent = 'Error: Could not connect to API.';
    }
  });

  // This function is injected into the webpage
  function getPageText() {
    // A simple way to grab all paragraph text
    const pTags = Array.from(document.querySelectorAll('p'));
    return pTags.map(p => p.textContent).join(' ');
  }
});