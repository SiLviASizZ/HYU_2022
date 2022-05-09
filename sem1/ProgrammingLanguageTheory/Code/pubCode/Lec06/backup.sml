
fun backup1 (f,g) = 
    fn x => case f x of
              NONE => g x
            | SOME y => y

fun backup2 (f, g) =
    fn x =>
        f x 
          handle _ => g x
