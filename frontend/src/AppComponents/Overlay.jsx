import React, { useState } from 'react';
import '../style.css';
import axios from 'axios';
import HomeOverlay from './HomeOverlay';
import SearchResultsOverlay from './SearchResultsOverlay';

export default function Overlay({setOverlayHide, setPlayback}) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [resultsLoaded, setResultsLoaded] = useState(false)
  const [suggestions, setSuggestions] = useState([]);

const handleSearch = async () => {
  try {
    const response = await axios.post('http://localhost:5000/search', { query });
    console.log('API Response:', response.data);
    if (response.status === 200) {
      setResults(response.data.results || []);
      setResultsLoaded(true);
      console.log('Results Loaded:', response.data.results);
      console.log(resultsLoaded)
    } else {
      console.error('Error fetching search results');
      setResultsLoaded(false);
    }
  } catch (error) {
    console.error('Error:', error);
  }
};

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleSearch();
      setPlayback(true)
      setOverlayHide(true)
    }
  };

  return (
    <>
      { !resultsLoaded && <HomeOverlay handleKeyPress={handleKeyPress} query={query} setQuery={setQuery}/> }
      { resultsLoaded && <SearchResultsOverlay results={results}/> }
    </>
  );
}
