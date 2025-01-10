import '../style.css'
import ReactDOM from 'react-dom/client'
import { Canvas, useThree } from '@react-three/fiber'
import Experience from '../SceneComponents/Experience.jsx'
import { Suspense, useEffect, useRef, useState } from 'react'
import Overlay from './Overlay.jsx'
import { useProgress } from '@react-three/drei'
import LoadingScreen from './LoadingScreen.jsx'
import { LoadingProvider } from '../Contexts/LoadingContext.jsx'

function CameraSetup() {
  const { camera } = useThree();

  useEffect(() => {
    // Set the camera's lookAt after the component is mounted
    camera.lookAt(0.36, 1.71, 0.33);
  }, [camera]);

  return null; // This component doesn't render anything
}

export default function App() {
    const [hide, setHideOverlay] = useState(false)
    const [playback, setPlayback] = useState(false);

    // Put audio stuff in its own component for reusability.
     const audioRef = useRef(null);

        // Initialize audio when component mounts
    useEffect(() => {
        if (audioRef.current) {
            // Add file extension if it's missing
            audioRef.current.src = "audio/pokemon_theme.mp3"; // or .wav, .ogg depending on your file
            audioRef.current.volume = 0.5; // Set initial volume
            
            // Auto-play handling (optional)
            const playAudio = () => {
                audioRef.current.play().catch(error => {
                    console.log("Audio autoplay failed:", error);
                    // Many browsers require user interaction before playing audio
                });
            };

        //     // Optional: Play on user interaction
        //     document.addEventListener('click', playAudio, { once: true });
            
        //     return () => {
        //         document.removeEventListener('click', playAudio);
        //     };
         }
    }, []);

    const handlePlay = () => {
        if (audioRef.current) {
            audioRef.current.play().catch(error => {
                console.log("Play failed:", error);
            });
        }
    };


  return (
    <>
    <LoadingProvider> 
    <LoadingScreen />  
    {!hide && <Overlay setOverlayHide={setHideOverlay} setPlayback={setPlayback}/>}
        <Canvas
        camera={ {
            fov: 45,
            near: 0.1,
            far: 200,
            position: [ 0.26, 1.77, 3.61 ],
        } }
    >
        <Suspense>
        <CameraSetup />
        <Experience overlayCallbackFn={setHideOverlay} playback={playback}/>
        </Suspense>
    </Canvas>
            {/* <div className="audio-controls"> */}
                {/* <audio 
                    ref={audioRef}
                    controls // Add controls if you want the default audio player UI
                    loop // Optional: if you want the audio to loop
                /> */}
                {/* Optional: Custom play button */}
                {/* <button className="absolute top-0 left-0" onClick={handlePlay}>Play Music</button> */}
            {/* </div>    */}
    </LoadingProvider>
    </>
  )
}
