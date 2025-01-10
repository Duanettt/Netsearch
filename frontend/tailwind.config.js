/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        pokemon_solid: ['PokemonSolid', 'sans-serif'],
        pokemon_hollow: ['PokemonHollow', 'sans-serif'],
        pokemon_pixel: ['Pixel', 'sans-serif']
      },
    },
  },
  plugins: [],
}