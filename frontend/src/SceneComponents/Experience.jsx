import { useFrame, useThree } from '@react-three/fiber'
import { OrbitControls, PresentationControls, Sky } from '@react-three/drei'
import { useEffect, useRef, useState } from 'react'
import { Perf } from 'r3f-perf'
import FBXModel from './FBXModel'
import GLTFModel from './GLTFModel'
import FBXModel2 from './FBXModel2'
import CameraPathControls from './CameraPathControls'


export default function Experience({overlayCallbackFn, checkHide, playback})
{
  const { camera, gl } = useThree();

  

  useEffect(() => {
    const handlePointerDown = () => {
      overlayCallbackFn(true)
    //   console.log('Pointer down detected globally');
    };

    const handlePointerUp = () => {
        overlayCallbackFn(false)
         //   console.log('Pointer up detected globally');
    };

    // Attach global event listeners
    gl.domElement.addEventListener('pointerdown', handlePointerDown);
    gl.domElement.addEventListener('pointerup', handlePointerUp);

    // Cleanup event listeners
    return () => {
      gl.domElement.removeEventListener('pointerdown', handlePointerDown);
      gl.domElement.removeEventListener('pointerup', handlePointerUp);
    };
  }, [gl]);

  useFrame(() => {

  })

    return <>
        <Sky></Sky>
        {/* <Perf position="top-right" /> */}

        {/* <OrbitControls camera={camera} makeDefault /> */}
        <CameraPathControls playback= {playback}/>
        <PresentationControls
        enabled={true} // the controls can be disabled by setting this to false
        global={false} // Spin globally or by dragging the model
        cursor={true} // Whether to toggle cursor style on drag
        snap={true} // Snap-back to center (can also be a spring config)
        speed={1} // Speed factor
        >
        <directionalLight position={ [ 1, 2, 3 ] } intensity={ 3.5 } />
        <ambientLight intensity={ 1.5 } />

        {/* <FBXModel model={'character/animations/Walking.fbx'} texturePath={'character/textures/face_texture.png'}/> */}
        <FBXModel2 />
        <GLTFModel path={'gameMap/pokemonCenter_f.glb'} />
        </PresentationControls>
    </>
}