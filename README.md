# Giicis
The repository of our blog


## Creating a new post 

If you want to create a new post, follow the next steps

1. Clone this repo `git clone https://github.com/giicis/giicis.github.io` 
2. Copy the template in `archetypes` and create a  new file in `content/posts/my-new-post.md`. In this template you should replace the placeholders with information about you and your post. After line 10 you can write your post in markdown format. Alternative, if you have hugo installed you can run `hugo new posts/your-new-post.md` and it will create a new post.
3. When you are finished, change `draft` to false.
4. Create a new PR to submit your changes


Also, you can test your changes locally if you have [Hugo](https://gohugo.io/) installed. Just run `hugo serve -D`


## Creating a new user

1. Inside the folder `content/members` copy `default.md` and create a new file named `your-name.md`.
2. Inside this new file, replace `title` with your name and write anything you want about yourself in markdown format.
3. Create a new PR to submit your changes