import { useGLTF } from '@react-three/drei'
import React, { useEffect } from 'react'

export default function GLTFModel({path}) {
    const model = useGLTF(path)

    useEffect(() => {
  if (model) {
    model.scene.traverse((child) => {
      if (child.isMesh) {
        console.log(child)
        child.castShadow = true;
        child.receiveShadow = true;
      }
    });
  }
}, [model]);

  return (
    <primitive scale={1}object={model.scene} />
  )
}
