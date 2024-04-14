const express=require('express')
const route=express.Router();
const {register,login,logout}=require('../controllers/Auth.Controller')

//Definition of Routes
route.post('/register',register)
route.post('/login',login)
route.post('/logout',logout)



module.exports=route