#!/bin/bash

#######################
# Auxiliary functions #
#######################
function init() {
	PROJECT_HOME=`git rev-parse --show-toplevel 2> /dev/null`
	status=$?
	if [ $status != 0 ]; then
		echo "Warning: you are not in a git repository."
		exit $status
	fi

	NUMBER_OF_FILES=10
	FILES_TO_REMOVE=$( mktemp )
	trap "rm $FILES_TO_REMOVE" EXIT

	echo "Scanning repository..."

	git gc --quiet
	deleted=0
}

function get_biggest_files_hash() {
	hashes=( `git verify-pack -v $PROJECT_HOME/.git/objects/pack/*.idx | sort -k 3 -n | tail -$1 | awk '{print $1}'` )
}

function get_biggest_files_size() {
	sizes=( `git verify-pack -v $PROJECT_HOME/.git/objects/pack/*.idx | sort -k 3 -n | tail -$1 | awk '{print $3}'` )
}

function get_file_name() {
	local file_name=`git rev-list --objects --all | grep $1 | awk '{print $2}'`
	echo "$file_name" 
}

function get_biggest_files_name() {
	files=()
	for hash in ${hashes[@]}; do
		files+=( `get_file_name $hash` )
	done
}

function verify_if_files_exist_now() {
	exists=()
	for file in ${files[@]}; do
		git ls-files | grep -w $file
		if [ $? -eq 0 ]; then
			exists+=( '*' )
		else
			exists+=( ' ' )
		fi
	done
}

function show_menu() {
	local options=()
	for ((i=0; i<$NUMBER_OF_FILES; i++)); do
		if [ $i -lt ${#files[@]} ]; then
			if [ "${exists[$i]}" == "*" ]; then
				options+=("${files[$i]}" "       ${sizes[$i]}${exists[$i]}   " "OFF")
			else
				options+=("${files[$i]}" "       ${sizes[$i]}${exists[$i]}   " "ON")
			fi
		fi
	done

	whiptail --title 'git delete' --checklist "Files with their size that might be removed. Files size marked with * exist at current state of the project." $(($NUMBER_OF_FILES + 12)) 90 $(($NUMBER_OF_FILES + 4)) \
		"${options[@]}" \
		2> $FILES_TO_REMOVE 

	clear
}

function delete_file() {
	git filter-branch -f --original "refs/original/$1" --index-filter "git rm --cached --ignore-unmatch $1" --prune-empty -- --all &> /dev/null
	echo "$1 removed"
	deleted=1
}

function delete_files() {
	echo "Deleting files..."
	for file in $( cat $FILES_TO_REMOVE ); do
		delete_file ${file//\"}
	done
}

function clean_backup_files() {
	if [ $deleted -eq 1 ]; then
		whiptail --title 'git delete' --yesno 'Clean your backup files in .git directory?' 8 78

		if [ $? -eq 0 ]; then
			echo "Cleaning backup files..."
			rm -rf $PROJECT_HOME/.git/refs/original/*
			git reflog expire --all --expire-unreachable=0 &> /dev/null
			git repack -A -d &> /dev/null
			git prune &> /dev/null
			echo "Backup files removed."
		fi
	fi
}

########
# main #
########
init
get_biggest_files_hash $NUMBER_OF_FILES
get_biggest_files_size $NUMBER_OF_FILES
get_biggest_files_name
verify_if_files_exist_now
show_menu
delete_files
clean_backup_files
