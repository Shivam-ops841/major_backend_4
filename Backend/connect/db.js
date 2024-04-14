const mongoose=require('mongoose')

const connect=async()=>{
    try {
        await mongoose.connect(process.env.MONGODB_URI)

        console.log('Connected to the database');
    } catch (error) {
        console.log(error)
    }
}

module.exports = connect