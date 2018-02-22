{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE ScopedTypeVariables #-}
module Main where
import Data.Array.Accelerate.LLVM.Native as CPU
import Data.Array.Accelerate
import Convolve(convFC)
import System.Environment(getArgs)

main :: IO ()
main = do
    args <- getArgs
    let samples = 1
    let inputSize = (Z :. samples :. 3 :. 2 :. 2)
    let xs = fromList inputSize [1..12]
    let kernelSize = Z :. 3 :. 3 :. 1 :. 1 :: DIM4
    let biasSize = Z :. 3 :. 2 :. 2 :: DIM3
    let kernel = fromList kernelSize ([1,0,-1,1,0,1,1,-1,1])
    let bias = CPU.run $ fill (constant biasSize) (0 :: Exp Double)
    let fc = CPU.runN convFC
    print (fc kernel bias xs)
    print (CPU.run $ convFC (use kernel) (use bias) (use xs))
