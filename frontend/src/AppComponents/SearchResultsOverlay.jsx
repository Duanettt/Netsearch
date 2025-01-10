import React from 'react';

export default function SearchResultsOverlay({ results }) {
  console.log(results)
  if (!results.length) return null;

  return (
    <div className="absolute top-0 left-0 w-full h-full flex justify-center items-center">
      <div className="bg-white/90 p-6 rounded-lg max-w-2xl max-h-[80vh] overflow-y-auto">
        {results.map((result, index) => (
          <div key={index} className="mb-4 border-b border-gray-200 pb-4 last:border-0">
            <h3 className="text-xl font-bold text-gray-800">{result}</h3>
            <p className="text-gray-600 mt-2">{result}</p>
          </div>
        ))}
      </div>
    </div>
  );
}