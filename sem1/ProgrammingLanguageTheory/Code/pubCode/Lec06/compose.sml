fun sqrt_of_abs i = Math.sqrt(Real.fromInt(abs i))
fun sqrt_of_abs i = (Math.sqrt o Real.fromInt o abs) i

val sqrt_of_abs =  Math.sqrt o Real.fromInt o abs

fun compose (f, g) = 
    fn x => f(g(x))

valsqrt_of_abs_longer = compose(Math.sqrt, compose(Real.fromInt, abs))
