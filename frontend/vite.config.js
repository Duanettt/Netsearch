import react from '@vitejs/plugin-react'
import { transformWithEsbuild } from 'vite'
import restart from 'vite-plugin-restart'

export default {
    root: 'src/',
    publicDir: '../public/',
    plugins: [
        // Restart server on static/public file change
        restart({ restart: [ '../public/**' ] }),

        // React support
        react(),

        // .js file support as if it was JSX
        {
            name: 'load+transform-js-files-as-jsx',
            async transform(code, id) {
                if (!id.match(/src\/.*\.js$/))
                    return null

                return transformWithEsbuild(code, id, {
                    loader: 'jsx',
                    jsx: 'automatic',
                });
            },
        },
    ],
    server: {
        host: true, // Open to local network and display URL
        open: false, // Prevent the browser from opening automatically
    },
    build: {
        outDir: '../dist', // Output in the dist/ folder
        emptyOutDir: true, // Empty the folder first
        sourcemap: true, // Add sourcemap
    },
}
