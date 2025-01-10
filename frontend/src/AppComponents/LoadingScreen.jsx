import '../style.css'
import React, { useState, useEffect } from 'react';
import { useProgress, Html } from '@react-three/drei';
import { useFrame } from '@react-three/fiber';
import { useLoading } from '../Contexts/LoadingContext';

const LoadingScreen = () => {
    const { progress } = useLoading()
    const [isLoaded, setIsLoaded] = useState(false);

    useEffect(() => {
        if (progress === 100) {
            setTimeout(() => setIsLoaded(true), 500);
        }
    }, [progress]);

    return (
            <div className="fixed inset-0 flex items-center justify-center bg-white">
                <div className="relative">
                    <div className={`
                        relative w-32 h-32 
                        ${isLoaded ? '' : 'animate-[spin_2s_linear_infinite]'}
                    `}>
                        {/* Top half */}
                        <div className={`
                            absolute top-0 left-0 w-32 h-16 
                            bg-red-600 rounded-t-full
                            transition-transform duration-1000
                            ${isLoaded ? 'translate-y-[-100vh]' : ''}
                        `}>
                            {/* Center button */}
                            <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-8 h-8 bg-white rounded-full border-4 border-black z-10" />
                        </div>
                        
                        {/* Bottom half */}
                        <div className={`
                            absolute top-16 left-0 w-32 h-16 
                            bg-white rounded-b-full border-t-4 border-black
                            transition-transform duration-1000
                            ${isLoaded ? 'translate-y-[100vh]' : ''}
                        `} />
                    </div>
                    
                    {/* Progress text */}
                    <div className="absolute top-full mt-4 left-1/2 -translate-x-1/2 text-black font-bold">
                        {Math.round(progress)}%
                    </div>
                </div>
            </div>
    );
};

export default LoadingScreen;