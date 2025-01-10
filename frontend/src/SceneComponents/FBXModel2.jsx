import { useLoader, useFrame } from '@react-three/fiber';
import React, { useEffect, useState } from 'react';
import { FBXLoader } from 'three/examples/jsm/loaders/FBXLoader';
import { AnimationMixer } from 'three';
import { a } from 'motion/react-client';
import { useModels } from '../CustomHooks/useModels';

export default function FBXModel2() {
    // Load the base model and animations
    const loadedModels = useModels()
    const baseModel = loadedModels.ashAnimations.idle
    const walkingAnim = loadedModels.ashAnimations.walking
    const victoryAnim = loadedModels.ashAnimations.victory
    const lookOverAnim = loadedModels.ashAnimations.lookover

    const [currentAction, setCurrentAction] = useState('idle');
    const [mixer] = useState(() => new AnimationMixer(baseModel));
    const [rotationY, setRotationY] = useState(0);
    const [pointer, setPointer] = useState({ x: 0, y: 0 });
    const [animations, setAnimations] = useState({});

    useEffect(() => {

        if (baseModel)
        {
            baseModel.castShadow = true
            baseModel.receiveShadow = true
        }
    }, [])

    // Initialize animations
    useEffect(() => {
        const newAnimations = {
            idle: mixer.clipAction(baseModel.animations[0]),
            walk: mixer.clipAction(walkingAnim.animations[0]),
            victory: mixer.clipAction(victoryAnim.animations[0]),
            lookover: mixer.clipAction(lookOverAnim.animations[0])
        };

        // Initialize all animations
        Object.values(newAnimations).forEach(anim => {
            anim.reset();
            anim.setEffectiveTimeScale(1);
            anim.setEffectiveWeight(1);
            anim.clampWhenFinished = true;
        });

        setAnimations(newAnimations);
        
        // Start with idle animation
        newAnimations.idle.play();

        return () => {
            Object.values(newAnimations).forEach(anim => anim.stop());
        };
    }, [baseModel, walkingAnim, victoryAnim, lookOverAnim, mixer]);

    // Handle animation transitions
    useEffect(() => {
        if (!animations[currentAction]) return;

        // Fade out all current animations
        Object.values(animations).forEach(anim => {
            if (anim.isRunning()) {
                anim.fadeOut(0.2);
            }
        });

        // Fade in the new animation
        const nextAnim = animations[currentAction];
        nextAnim.reset().fadeIn(0.2).play();

    }, [currentAction, animations]);

    const handleKeyDown = (event) => {
        switch (event.key) {
            case 'w':
                setCurrentAction('walk');
                break;
            case 'i':
                setCurrentAction('idle');
                break;
            case 'r':
                setCurrentAction('victory');
                break;
            case 'b':
                setCurrentAction('lookover');
                break;
        }
    };

    useEffect(() => {
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, []);

    const handlePointerMove = (event) => {
        const x = (event.clientX / window.innerWidth) * 2 - 1;
        const y = -(event.clientY / window.innerHeight) * 2 + 1;
        setPointer({ x, y });
    };

    useEffect(() => {
        window.addEventListener('pointermove', handlePointerMove);
        return () => window.removeEventListener('pointermove', handlePointerMove);
    }, []);

    useFrame((state, delta) => {
        mixer.update(delta);
        
        const targetRotation = pointer.x * Math.PI * 0.1;
        const rotationSpeed = 0.1;
        setRotationY(targetRotation);
    });

    if (!baseModel || !walkingAnim || !victoryAnim || !lookOverAnim) return null;

    return (
        <primitive 
            object={baseModel} 
            scale={0.005} 
            position={[1.0, 0.0, 2.0]}
            rotation={[0, rotationY, 0]} 
        />
    );
}