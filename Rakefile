# frozen_string_literal: true

require 'date'
require 'yaml'
require 'fileutils'

# The root directory of posts
POST_ROOT = '_posts'
# The directory for post images
IMAGE_DIR = '/assets/img/posts'

namespace :post do
  desc 'Create a new post.'
  task :new do
    # Ensure the posts directory exists
    FileUtils.mkdir_p(POST_ROOT)

    print 'Title: '
    title = STDIN.gets.chomp

    if title.empty?
      puts 'Error: Title cannot be empty.'
      exit 1
    end

    slug = title.downcase.strip.gsub(' ', '-').gsub(/[^\w-]/, '')
    today = Date.today.to_s
    filename = "#{POST_ROOT}/#{today}-#{slug}.md"

    if File.exist?(filename)
      puts "Error: Post '#{filename}' already exists."
      exit 1
    end

    # Default front matter
    front_matter = {
      'title' => title,
      'date' => DateTime.now.strftime('%Y-%m-%d %H:%M:%S %z'),
      'categories' => [],
      'tags' => [],
      'pin' => false,
      'toc' => true,
      'img_path' => "#{IMAGE_DIR}/#{today.gsub('-', '/')}"
    }.to_yaml

    content = <<~CONTENT
      #{front_matter}---

      <!--
      The post content starts here.
      -->

    CONTENT

    File.write(filename, content)
    puts "Successfully created post: '#{filename}'"
    puts 'Please edit the file to add categories, tags, and content.'
  end
end 