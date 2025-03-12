import numpy as np
import heapq


def update_alignments(alignments, alignment_probs, top_tokens, top_probs, beam_width=3, blank_index=0):

    if len(alignments) == 0:
        alignments.extend([[i] for i in top_tokens])
        alignment_probs.extend(top_probs)
        return alignments, alignment_probs


    new_alignments = []
    new_alignment_probs = []

    for ind, alignment in enumerate(alignments):
        last_char = alignment[-1]
        for token, prob in zip(top_tokens, top_probs):
            if token == last_char: # If it's the same as before (whether blank or repeated char - it gets collapsed)
                new_alignment = alignment
            elif last_char == blank_index:  # If previous is a blank and this is a character, we can get rid of the previous blank
                new_alignment = alignment[:-1] + [token]
            else:
                new_alignment = alignment + [token]
            
            if new_alignment in new_alignments:
                change_index = new_alignments.index(new_alignment)
                old_prob = new_alignment_probs[
                    change_index]
                new_prob = np.log(np.exp(old_prob) + np.exp(alignment_probs[ind] + prob))
                new_alignment_probs[
                    change_index] += new_prob
            else:
                new_alignments.append(new_alignment)
                new_alignment_probs.append(alignment_probs[ind] + prob)

    # return the most probable one
    # and then reduce to the beam width
    # Sort new_alignment_probs in reverse order while preserving the relative order of new_alignments
    sorted_pairs = sorted(zip(new_alignment_probs, new_alignments), key=lambda x: -x[0])

    # Unzip the sorted result
    new_alignment_probs, new_alignments = zip(*sorted_pairs)

    # Convert back to lists if needed
    new_alignment_probs = list(new_alignment_probs)
    new_alignments = list(new_alignments)

    return new_alignments[:beam_width], new_alignment_probs[:beam_width]


def beam_search_ctc(prob_matrix, beam_width=3, blank=0, n_classes=17, return_alignments=False):
    
    # Get top n probabilities and their corresponding indices for each time step
    # Create a list of alignments sequentially, collapsing and combining as you go
    indices = np.arange(n_classes)
    alignments, alignment_probs = [], []
    for ind, probs in enumerate(prob_matrix):
        # Get the top 3
        # previous_alignments adding - collapse at will - if the same as previous, don't add 
        # If new and the previous is blank, remove the blank
        top_n = heapq.nlargest(n_classes, enumerate(probs), key=lambda x: x[1])
        top_tokens = [i[0] for i in top_n]
        top_probs = [float(i[1]) for i in top_n]
        alignments, alignment_probs = update_alignments(
            alignments, alignment_probs, top_tokens, top_probs,
            beam_width=beam_width)
        
    if return_alignments:
        return alignments
    return " ".join([str(i) for i in alignments[0]])