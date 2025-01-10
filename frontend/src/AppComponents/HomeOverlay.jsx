import React from 'react'
import '../style.css';

export default function HomeOverlay({query, setQuery, handleKeyPress}) {
  return (
    <div className="absolute top-1/4 right-1/6 pr-24 w-full flex flex-col justify-center items-center text-white z-10">
      <img src="fonts/NETSEARCH.png" alt="NetSearch Logo" className="mb-4" />
      <div className="flex items-center">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Search with Netsearch for games of your choice..."
          className="search_bar"
        />
      </div>
    </div>
  );

}
