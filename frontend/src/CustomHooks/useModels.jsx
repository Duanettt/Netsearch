// Responsible for loading up our models and keeping a centralized system

import { useLoader } from "@react-three/fiber"
import { useLoading } from "../Contexts/LoadingContext"
import { FBXLoader, GLTFLoader } from "three/examples/jsm/Addons.js"


export const useModels = () =>
{
    const { manager } = useLoading()

    const pokemonCenter = useLoader(GLTFLoader,
        'gameMap/pokemonCenter_f.glb',
        (loader) => {
            loader.manager = manager
        }
    )
    const ashIdleAnimation = useLoader(FBXLoader,
        'character/animations/Idle.fbx',
        (loader) => {
            loader.manager = manager
        }
    )
    const ashWalkingAnimation = useLoader(FBXLoader,
        'character/animations/Walking.fbx',
        (loader) => {
            loader.manager = manager
        }
    )
    const ashVictoryAnimation = useLoader(FBXLoader,
        'character/animations/Victory_Idle.fbx',
        (loader) => {
            loader.manager = manager
        }
    )

    const ashLookOverAnimation = useLoader(FBXLoader,
        'character/animations/Look_Over_Shoulder.fbx'
    )

    return {
        pokemonCenter,
        ashAnimations:
        {
            idle: ashIdleAnimation,
            walking: ashWalkingAnimation,
            victory: ashVictoryAnimation,
            lookover: ashLookOverAnimation,
        }
    }

}