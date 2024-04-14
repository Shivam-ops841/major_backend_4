const express=require('express')
const app=express()
require('dotenv').config()
const connect=require('./connect/db')

//Importing of routes defined by me
const Auth_Routes=require('./routes/Auth.Routes')
const User_Routes=require('./routes/User.Routes')


//Using of middlewares
app.use(express.json())
app.use(express.urlencoded({extended:true}))

//Importing Of Routes
app.use('/api/v1/auth',Auth_Routes)
app.use('/api/v1/user',User_Routes)

connect().then(()=>{
    app.listen(process.env.PORT,()=>{
        console.log(`Listening on ${process.env.PORT}`)
    })
}).catch(err=>console.log(err))