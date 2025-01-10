import { useAnimations, useFBX, useTexture } from '@react-three/drei';
import React, { useEffect, useState } from 'react';

export default function FBXModel({ model, texturePath, emissionPath, lightMapPath }) {
    const ashWalking = useFBX('character/animations/Walking.fbx'); // Load the FBX model
    const ashIdle = useFBX('character/animations/Idle.fbx');
    const modelOptions = [ashWalking, ashIdle];

    const texture = useTexture(texturePath); // Load the texture
    // const emissionTexture = useTexture(emissionPath);
    // const lightMapTexture = useTexture(lightMapPath);
    const [renamed, setRenamed] = useState(false);
    const [selectedModel, setSelectedModel] = useState(null); // Initialize as null

    // We use use effect alot to be fair but its the first render and allows us to set up things before we render to our experience.
    useEffect(() => {
        // Once we

        if (ashWalking && ashIdle) {
            setSelectedModel(ashIdle); // Set default model (walking)
        }
    }, [ashWalking, ashIdle]);

    useEffect(() => {
        if (selectedModel && texture) {
            // Traverse the FBX model to apply the texture
            selectedModel.traverse((child) => {
                if (child.isMesh) {
                }
            });
        }
    }, [selectedModel, texture]);

    // Rename animations if needed
    useEffect(() => {
        if (selectedModel) {
            selectedModel.animations.forEach((clip) => {
                if (clip.name === "mixamo.com") {
                    clip.name = "Animation"; // Rename "mixamo.com" to "Animation"
                }
            });
            setRenamed(true); // Signal that renaming is complete
        }
    }, [selectedModel]);

    // Use animations only after renaming
    const animations = useAnimations(renamed ? ashIdle.animations : [], ashIdle);
    const animations2 = useAnimations(renamed ? ashWalking.animations : [], ashWalking);

    useEffect(() => {
        if (renamed && animations.actions?.Animation) {
            animations.actions.Animation.fadeIn(0.5).play();
        }
        
    }, [renamed, animations.actions]);


    // This is crucial until the models are loaded we return null to our function to not render anything.
    if (!selectedModel || !texture) return null; 

    return (
        <primitive object={selectedModel} scale={0.005} position={[1.0, 0.0, 2.0]} />
    );
}
