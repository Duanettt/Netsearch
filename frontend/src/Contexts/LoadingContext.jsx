import React, { createContext, useContext, useState } from 'react';
import { LoadingManager } from 'three';

const LoadingContext = createContext()

export const LoadingProvider = ({ children }) => {
  const [progress, setProgress] = useState(0);
  const [isLoaded, setIsLoaded] = useState(false);
  const [totalItems, setTotalItems] = useState(0);
  const [loadedItems, setLoadedItems] = useState(0);

  // Create loading manager
  const manager = new LoadingManager();

  manager.onStart = (url, loaded, total) => 
  {
    console.log(loaded)
  }

  manager.onProgress = (url, loaded, total) => {
    setLoadedItems(loaded);
    setTotalItems(total);
    const currentProgress = (loaded / total) * 100;
    // console.log(currentProgress)
    setProgress(currentProgress);
  };

  manager.onLoad = () => {
    setProgress(100);
    setIsLoaded(true);
  };

  manager.onError = (url) => {
    console.error('Error loading:', url);
  };

  const value = {
    manager,
    progress,
    isLoaded,
    totalItems,
    loadedItems
  };

  return (
    <LoadingContext.Provider value={value}>
      {children}
    </LoadingContext.Provider>
  );
};

export const useLoading = () => {
    const context = useContext(LoadingContext)

    if (!context)
    {
        throw new Error('useLoading must be initialized with a LoadingProvider, LoadingProvider allows to encapsulate and instantiate a context in which the components below it can use this.')
    }
    return context
}

