import React, { useRef, useState, useEffect } from 'react';
import { useThree, useFrame } from '@react-three/fiber';
import { CatmullRomCurve3, Vector3 } from 'three';
import { PivotControls } from '@react-three/drei';


const defaultInitialPoints = [
  [0.26, 1.76, 3.71],
  [0.00, 2.21, 0.83],
  [0.00, 2.25, -0.01],
];

export default function CameraPathControls({ 
  initialPoints = defaultInitialPoints,
  showControls = true,
  curveColor = "yellow",
  pointColor = "red",
  addPointColor = "green",
  playbackColor = "blue",
  animationSpeed = 0.0015,
  onPathUpdate = null,
  playback = false
  
}) {
  // Rest of the code remains the same
  const [controlPoints, setControlPoints] = useState(
    initialPoints.map(([x, y, z]) => new Vector3(x, y, z))
  );
  
  const [curve, setCurve] = useState(null);
  const progress = useRef(0);
  const { camera } = useThree();

  useEffect(() => {
    const newCurve = new CatmullRomCurve3(controlPoints, false);
    setCurve(newCurve);
    if (onPathUpdate) onPathUpdate(newCurve);
  }, [controlPoints]);
  
  useFrame(() => {
    if (playback && curve) {
      progress.current += animationSpeed;
      if (progress.current > 0.7) {
        progress.current = 0.7;
      }
      const point = curve.getPoint(progress.current);
      camera.position.copy(point);
      
      const lookAtPoint = curve.getPoint(Math.min(progress.current + 0.01, 1));
      camera.lookAt(lookAtPoint);
    }
  });

  if (!showControls) {
    return null;
  }

  return (
    <group>
      <mesh
        position={[0, 3, 0]}
        onClick={() => {
          setControlPoints([
            ...controlPoints,
            new Vector3(0, 2, 0)
          ]);
        }}
      >
        <sphereGeometry args={[0.2]} />
        <meshBasicMaterial color={addPointColor} />
      </mesh>
      <mesh
        position={[2, 3, 0]}
      >
        <boxGeometry args={[0.3, 0.3, 0.3]} />
        <meshBasicMaterial color={playback ? "red" : playbackColor} />
      </mesh>
    </group>
  );
}